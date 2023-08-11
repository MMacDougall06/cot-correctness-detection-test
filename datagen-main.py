import os
import time
import openai
import logging
from datasets import load_dataset as hfds_load
import textwrap
import csv
import random
from fractions import Fraction
from itertools import chain
import re

openai.api_key = None
if openai.api_key is None:
    raise Exception('Open AI API key required')

def complete(prompt, model='text-davinci-003'):
    err_count = 0
    while True:
        try:
            time.sleep(0.5) # send at most 2 rps across different calls
            completion = openai.Completion.create(
                model=model,
                prompt=prompt,
                stop=None,
                temperature=0.,
                max_tokens=512,
                top_p=1.,
                frequency_penalty=0.,
                presence_penalty=0.,
                n=1,
                best_of=1
            )
            return completion['choices'][0]['text']
        except openai.error.OpenAIError as e:
            logging.exception(f'OpenAI error: {e}')
            err_count += 1
            if err_count > 20:
                raise
            time.sleep(5.)

def escape_newlines(x):
    if isinstance(x, str):
        return x.replace('\n', r'\n')
    return x

non_numeric_re = re.compile(r'[^0-9.\/]')
def to_float(s):
    global non_numeric_re
    s = non_numeric_re.sub('', s)
    if '/' in s:
        return float(Fraction(s))
    return float(s)

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
        "argument": "''')

QUESTION_PROMPT_ANSWER_FIRST = textwrap.dedent('''\
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    A question is given in the following JSON format:
    {{
        "question": "(question goes here)"
    }}
    Write a step by step argument to answer the question. Provide your response in the following JSON format:
    {{
        "answer": "(final answer number goes here)",
        "argument": "(step by step argument goes here)"
    }}

    ### Input:
    {{
        "question": "{}"
    }}

    ### Response:
    {{
        "answer": "''')

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
        "argument": "''')

generations_checked = 0
generations_bad_format = 0

def get_right_answer(in_question, in_answer):
    global generations_checked
    global generations_bad_format
    question_prompt = QUESTION_PROMPT_ARGUMENT_FIRST.format(in_question)
    generated_answer = complete(question_prompt)
    answer_pieces = generated_answer.split('"')
    generations_checked += 1
    if len(answer_pieces) != 6:
        if len(answer_pieces) == 4:
            # probably didn't enquote the number...
            try:
                final_answer = to_float(answer_pieces[3])
            except ValueError:
                generations_bad_format += 1
                return None
        else:
            generations_bad_format += 1
            return None
    else:
        try:
            final_answer = to_float(answer_pieces[4])
        except ValueError:
            generations_bad_format += 1
            return None
    argument = answer_pieces[0].strip()
    is_correct = final_answer == to_float(in_answer)
    if is_correct:
        return str(final_answer), argument
    return None

def get_wrong_answer(in_question, in_answer):
    global generations_checked
    global generations_bad_format
    question_prompt = QUESTION_PROMPT_ANSWER_FIRST.format(in_question)
    generated_answer = complete(question_prompt)
    answer_pieces = generated_answer.split('"')
    generations_checked += 1
    if len(answer_pieces) != 6:
        generations_bad_format += 1
        return None
    try:
        final_answer = to_float(answer_pieces[0])
    except ValueError:
        generations_bad_format += 1
        return None
    argument = answer_pieces[4].strip()
    is_correct = final_answer == to_float(in_answer)
    if not is_correct:
        return str(final_answer), argument
    return None

def main():
    global generations_checked
    global generations_bad_format
    DATASET_NAMES = {
        'addsub.json',
        'singleop.json',
        'svamp_structured.json',
        'multiarith.json',
        'asdiv.json',
        'GSM8k_structured.json',
        'singleq.json',
        'simuleq.json',
    }
    lila_data = hfds_load('allenai/lila', 'iid') \
        .filter(lambda x: x['dataset'] in DATASET_NAMES) \
        .shuffle(seed=1)
    # bigbench_data = hfds_load('tasksource/bigbench','elementary_math_qa')

    entries_output = 0

    start_time = time.time()

    LOG_INTERVAL = 1

    OUTPUT_DIR = 'data/'
    OUTPUT_FILE = 'dataset.csv'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_DIR + OUTPUT_FILE, 'a', newline='') as outfile:
        outwriter = csv.writer(outfile)
        outwriter.writerow(['Question', 'Correct Answer', 'Correct Argument', 'Incorrect Answer', 'Incorrect Argument', 'Choice Answer', 'Choice Argument', 'Choice Answers Swapped'])

        fail_count = 0
        for i, entry in enumerate(lila_data['train']):
            try:
                correct_answer = get_right_answer(entry['input'], entry['output_answer'])
                if correct_answer is None:
                    continue

                incorrect_answer = get_wrong_answer(entry['input'], entry['output_answer'])
                if incorrect_answer is None:
                    continue

                # generate the answer choice arguments
                swap_answer_order = random.random() < 0.5
                swapped_answers = (correct_answer[1] + ' Final answer: ' + correct_answer[0], incorrect_answer[1] + ' Final answer: ' + incorrect_answer[0])[::-1 if swap_answer_order else 1]
                choice_prompt = CHOOSE_ANSWER_PROMPT.format(entry['input'], swapped_answers[0], swapped_answers[1])

                generated_answer = complete(choice_prompt)
                answer_pieces = generated_answer.split('"')
                generations_checked += 1
                if len(answer_pieces) != 6:
                    generations_bad_format += 1
                    continue
                generated_choice = answer_pieces[4].strip()
                if generated_choice != 'candidate_1' and generated_choice != 'candidate_2':
                    generations_bad_format += 1
                    continue
                argument = answer_pieces[0].strip()
                answer_choice = (generated_choice, argument, swap_answer_order)

                new_output = [escape_newlines(s) for s in chain((entry['input'],), correct_answer, incorrect_answer, answer_choice)]
                entries_output += 1

                outwriter.writerow(new_output)
            except Exception:
                fail_count += 1
                logging.exception(f'Generation failed {fail_count}')
            finally:
                if ((i+1) % LOG_INTERVAL) == 0:
                    logging.info(f'{(i+1)}: time {int(time.time() - start_time)} ent_o {entries_output} gen_c {generations_checked} gen_b {generations_bad_format}')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
