from huggingface.modeling_llama import LlamaForCausalLM #, LlamaConfig
from peft import prepare_model_for_kbit_training, PeftModel #, LoraConfig, get_peft_model
from datasets import load_dataset as hfds_load
import torch
import torch.nn as nn
import torch.nn.functional as F
import textwrap
import random
from transformers import LlamaTokenizerFast
from itertools import chain
import numpy as np
import logging
import time

np.random.seed(1)
torch.random.manual_seed(1)
random.seed(1)

DEVICE = torch.device('cuda')

SUBBATCH_SZ = 2
BATCH_SZ = (128 // SUBBATCH_SZ) * SUBBATCH_SZ
TEST_BATCH_SZ = 64
TEST_INTERVAL = 128 // SUBBATCH_SZ

ONE_TOKEN_ID = 29896
TWO_TOKEN_ID = 29906

INPUT_DIR = 'data'

def main():
    base_model = None
    def create_model():
        nonlocal base_model
        if base_model is None:
            from transformers import BitsAndBytesConfig
            # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False)
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model = LlamaForCausalLM.from_pretrained('decapoda-research/llama-7b-hf', quantization_config=quantization_config, device_map='auto', torch_dtype=torch.bfloat16)
            base_model = prepare_model_for_kbit_training(base_model)

            # model_config = LlamaConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=64)
            # base_model = LlamaForCausalLM(model_config).to(device=DEVICE, dtype=torch.bfloat16)

        # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        # lora_rank = 64
        # lora_alpha = 32
        # lora_dropout = 0.05
        # lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, bias='none', task_type='CAUSAL_LM')
        # return get_peft_model(model, lora_config).to(DEVICE)

        model = PeftModel.from_pretrained(base_model, 'tloen/alpaca-lora-7b', torch_dtype=torch.bfloat16, is_trainable=True)
        return model

    tokenizer: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained('huggyllama/llama-7b')
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    arguer_model: LlamaForCausalLM = create_model()
    chooser_model: LlamaForCausalLM = create_model()

    opt = torch.optim.AdamW(nn.ModuleList([arguer_model, chooser_model]).parameters(), lr=3e-4 * BATCH_SZ / 128, betas=(0.9,0.999), eps=1e-16, weight_decay=0.)

    QUESTION_PROMPT_ARGUMENT_FIRST = textwrap.dedent('''\
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        A question is given in the following JSON format:
        {{
            "question": "(question goes here)"
        }}
        Write a step by step argument to answer the question. Provide your response in the following JSON format:
        {{
            "argument": "(step by step argument goes here)",
            "answer": "(final answer number goes here)"
        }}

        ### Input:
        {{
            "question": "{}"
        }}

        ### Response:
        {{
            "argument": "{}''')

    def get_argument_kvs(questions, arguments):
        input_texts = []
        for question, argument in zip(questions, arguments):
            input_texts.append(QUESTION_PROMPT_ARGUMENT_FIRST.format(question, fence_text(argument)))
        encoded = tokenizer.__call__(input_texts, padding='longest', return_tensors='pt')
        input_ids, fence_inds = remove_tokenization_fences(encoded['input_ids'].to(DEVICE))
        # subtract one so that the inserted Ks and Vs line up with the inserted tokens being generated rather than the tokens themselves
        copy_ranges = (fence_inds - 1 - torch.arange(fence_inds.shape[1], device=DEVICE).unsqueeze(0)).unsqueeze(1)
        transplant_dict = {
            'ranges_to_copy': copy_ranges,
            'out_keys': [([],) for _ in range(len(questions))],
            'out_vals': [([],) for _ in range(len(questions))],
        }
        attention_mask = encoded['attention_mask'].to(DEVICE)[:,:-fence_inds.shape[1]]
        arguer_model.forward(input_ids=input_ids, attention_mask=attention_mask, transplant_dict=transplant_dict, use_cache=False)
        return {
            'arguments': arguments,
            'keys': transplant_dict['out_keys'],
            'vals': transplant_dict['out_vals'],
        }

    def fence_text(in_str, fence_token='[SEP]'):
        return fence_token + in_str + fence_token
    def remove_tokenization_fences(token_ids, fence_id=32000):
        is_fence = token_ids == fence_id
        # assumes the same number of fences for each sequence
        return token_ids[torch.logical_not(is_fence)].view(token_ids.shape[0], -1), is_fence.nonzero(as_tuple=True)[-1].view(token_ids.shape[0], -1)

    CHOOSE_ANSWER_PROMPT = textwrap.dedent('''\
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        A question and two answer candidates are given in the following JSON format:
        {{
            "question": "(question goes here)",
            "candidate_1": "(answer candidate 1 goes here)",
            "candidate_2": "(answer candidate 2 goes here)"
        }}
        Analyze the answer candidates and argue which is the best and why. Provide your response in the following JSON format:
        {{
            "argument": "(argument goes here)",
            "answer_choice": ("candidate_1" or "candidate_2")
        }}

        ### Input:
        {{
            "question": "{}",
            "candidate_1": {},
            "candidate_2": {}
        }}

        ### Response:
        {{
            "argument": "{}",
            "answer_choice": "candidate_''')

    def get_choice_logits(questions, correct_answers, incorrect_answers, choice_arguments, swap_answers):
        input_texts = []
        for question, answer_1, answer_2, choice_argument, swap_answer in zip(questions, correct_answers['arguments'], incorrect_answers['arguments'], choice_arguments, swap_answers):
            if swap_answer:
                answer_1, answer_2 = answer_2, answer_1
            input_texts.append(CHOOSE_ANSWER_PROMPT.format(question, fence_text(answer_1), fence_text(answer_2), choice_argument))
        encoded = tokenizer.__call__(input_texts, padding='longest', return_tensors='pt')
        input_ids, fence_inds = remove_tokenization_fences(encoded['input_ids'].to(DEVICE))
        attention_mask = encoded['attention_mask'].to(DEVICE)[:,:-fence_inds.shape[1]]
        # subtract one so that the inserted Ks and Vs line up with the inserted tokens being generated rather than the tokens themselves
        replace_ranges = (fence_inds - 1 - torch.arange(fence_inds.shape[1], device=DEVICE).unsqueeze(0)).view(len(questions), -1, 2)
        transplant_dict = {
            'ranges_to_replace': replace_ranges,
            'in_keys': [[iter(r) for r in (chain(a2_ranges, a1_ranges) if swap_answer else chain(a1_ranges, a2_ranges))] for a1_ranges, a2_ranges, swap_answer in
                            zip(correct_answers['keys'], incorrect_answers['keys'], swap_answers)],
            'in_vals': [[iter(r) for r in (chain(a2_ranges, a1_ranges) if swap_answer else chain(a1_ranges, a2_ranges))] for a1_ranges, a2_ranges, swap_answer in
                            zip(correct_answers['vals'], incorrect_answers['vals'], swap_answers)],
        }
        ans_out = chooser_model.forward(input_ids=input_ids, attention_mask=attention_mask, transplant_dict=transplant_dict, use_cache=False)
        return ans_out.logits[:,-1]
    
    start_time = time.time()
    train_num = 0
    test_num = 0
    epoch = 0

    data = hfds_load(INPUT_DIR, keep_in_memory=True)['train'] \
            .train_test_split(test_size=0.1, seed=1, keep_in_memory=True)

    opt.zero_grad()

    def train(i, entries):
        nonlocal train_num
        train_num += 1
        correct_kvs = get_argument_kvs(entries['Question'], entries['Correct Argument'])
        incorrect_kvs = get_argument_kvs(entries['Question'], entries['Incorrect Argument'])
        correct_token_indices = torch.tensor([TWO_TOKEN_ID if s else ONE_TOKEN_ID for s in entries['Choice Answers Swapped']], device=DEVICE, dtype=torch.int64)

        correct_final_logits = get_choice_logits(entries['Question'], correct_kvs, incorrect_kvs, entries['Choice Argument'], entries['Choice Answers Swapped'])
        correct_final_logits_softmax = F.log_softmax(correct_final_logits, dim=-1)
        loss_correct = -torch.diagonal(correct_final_logits_softmax.index_select(dim=1, index=correct_token_indices)).mean()
        loss_correct.backward()

        loss_incorrect = torch.zeros((1,), device=DEVICE)

        if (i + 1) % (BATCH_SZ // SUBBATCH_SZ) == 0:
            opt.step()
            opt.zero_grad()

        answers_swapped = torch.tensor(entries['Choice Answers Swapped'], device=DEVICE, dtype=torch.bool)
        correct_answer_chosen =  torch.logical_xor(correct_final_logits_softmax[:,ONE_TOKEN_ID] > correct_final_logits_softmax[:,TWO_TOKEN_ID], answers_swapped)
        original_choice_correct = torch.tensor([(a == 'candidate_1') ^ s for a, s in zip(entries['Choice Answer'], entries['Choice Answers Swapped'])], device=DEVICE, dtype=torch.bool)
        fr_correct = correct_answer_chosen.float().mean()
        fr_correct_when_orig_correct = (correct_answer_chosen * original_choice_correct).float().mean() / (original_choice_correct.float().mean().clamp_min_(1))
        fr_correct_when_orig_incorrect = (correct_answer_chosen * torch.logical_not(original_choice_correct)).float().mean() / (torch.logical_not(original_choice_correct).float().mean().clamp_min_(1))
        odds_diff = torch.where(answers_swapped,
                correct_final_logits_softmax[:,TWO_TOKEN_ID].exp() - correct_final_logits_softmax[:,ONE_TOKEN_ID].exp(),
                correct_final_logits_softmax[:,ONE_TOKEN_ID].exp() - correct_final_logits_softmax[:,TWO_TOKEN_ID].exp())
        odds_diff_correct = (odds_diff * correct_answer_chosen).mean() / correct_answer_chosen.float().mean().clamp_min_(1)
        odds_diff_incorrect = (odds_diff.abs_() * torch.logical_not(correct_answer_chosen)).mean() / torch.logical_not(correct_answer_chosen).float().mean().clamp_min_(1)
        logging.info(f' {epoch}-{train_num}: t {int(time.time() - start_time)} l_c {loss_correct.item():.3} l_i {loss_incorrect.item():.3} f_c {fr_correct.item():.3} ' +
                f'f_c_oc {fr_correct_when_orig_correct.item():.3} f_c_oi {fr_correct_when_orig_incorrect.item():.3} o_c {odds_diff_correct.item():.3} o_i {odds_diff_incorrect.item():.3}')

    @torch.inference_mode()
    def test():
        nonlocal test_num
        test_num += 1
        loss_correct = torch.zeros((1,), device=DEVICE)
        loss_incorrect = torch.zeros((1,), device=DEVICE)
        test_stats = torch.zeros((5,), device=DEVICE)
        num_stat_batches = torch.zeros((5,), device=DEVICE)
        for entries in data['test'].iter(TEST_BATCH_SZ):
            correct_kvs = get_argument_kvs(entries['Question'], entries['Correct Answer'], entries['Correct Argument'])
            incorrect_kvs = get_argument_kvs(entries['Question'], entries['Incorrect Answer'], entries['Incorrect Argument'])
            correct_token_indices = torch.tensor([TWO_TOKEN_ID if s else ONE_TOKEN_ID for s in entries['Choice Answers Swapped']], device=DEVICE, dtype=torch.int64)

            correct_final_logits = get_choice_logits(entries['Question'], correct_kvs, incorrect_kvs, entries['Choice Argument'], entries['Choice Answers Swapped'])
            correct_final_logits_softmax = F.log_softmax(correct_final_logits, dim=-1)
            loss_correct += -torch.diagonal(correct_final_logits_softmax.index_select(dim=1, index=correct_token_indices)).mean()

            answers_swapped = torch.tensor(entries['Choice Answers Swapped'], device=DEVICE, dtype=torch.bool)
            correct_answer_chosen =  torch.logical_xor(correct_final_logits_softmax[:,ONE_TOKEN_ID] > correct_final_logits_softmax[:,TWO_TOKEN_ID], answers_swapped)
            original_choice_correct = torch.tensor([(a == 'candidate_1') ^ s for a, s in zip(entries['Choice Answer'], entries['Choice Answers Swapped'])], device=DEVICE, dtype=torch.bool)
            test_stats[0] += correct_answer_chosen.float().mean()
            num_stat_batches[0] += 1
            if original_choice_correct.float().mean().item() > 0:
                test_stats[1] += (correct_answer_chosen * original_choice_correct).float().mean() / original_choice_correct.float().mean()
                num_stat_batches[1] += 1
            if torch.logical_not(original_choice_correct).float().mean().item() > 0:
                test_stats[2] += (correct_answer_chosen * torch.logical_not(original_choice_correct)).float().mean() / torch.logical_not(original_choice_correct).float().mean()
                num_stat_batches[2] += 1
            odds_diff = torch.where(answers_swapped,
                    correct_final_logits_softmax[:,TWO_TOKEN_ID].exp() - correct_final_logits_softmax[:,ONE_TOKEN_ID].exp(),
                    correct_final_logits_softmax[:,ONE_TOKEN_ID].exp() - correct_final_logits_softmax[:,TWO_TOKEN_ID].exp())
            if correct_answer_chosen.float().mean().item() > 0:
                test_stats[3] += (odds_diff * correct_answer_chosen).mean() / correct_answer_chosen.float().mean()
                num_stat_batches[3] += 1
            if torch.logical_not(correct_answer_chosen).float().mean().item() > 0:
                test_stats[4] += (odds_diff.abs_() * torch.logical_not(correct_answer_chosen)).mean() / torch.logical_not(correct_answer_chosen).float().mean()
                num_stat_batches[4] += 1
        test_stats /= num_stat_batches.clamp_min_(1)
        loss_correct /= num_stat_batches[0]
        loss_incorrect /= num_stat_batches[0]
        logging.info(f' test {test_num}: t {int(time.time() - start_time)} l_c {loss_correct.item():.3} l_i {loss_incorrect.item():.3} f_c {test_stats[0].item():.3} ' +
                f'f_c_oc {test_stats[1].item():.3} f_c_oi {test_stats[2].item():.3} o_c {test_stats[3].item():.3} o_i {test_stats[4].item():.3}')

    while True:
        epoch += 1
        for i, entries in enumerate(data['train'].iter(SUBBATCH_SZ)):
            try:
                train(i, entries)
            except Exception as e:
                if 'out of memory' not in str(e):
                    raise
                logging.exception(f'Ran out of memory, skipping sub-batch {i}')
            if (i + 1) % TEST_INTERVAL == 0:
                test()

if __name__ == '__main__':
    import sys
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler('run_logs/log.txt')
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logging.getLogger().addHandler(sh)
    main()
