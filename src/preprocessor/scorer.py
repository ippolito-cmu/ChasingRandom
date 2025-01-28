# emulating the https://github.com/gpt4life/alpagasus/blob/main/rating/chatgpt_rating.py unofficial but approved implementation in the prompt as closely as possible. 
import subprocess
import tqdm
import json
import openai
from time import sleep
import argparse
import time 
from openai import OpenAI
import os

client = OpenAI()
openai.api_key = os.getenv('OPENAI_API_KEY')

def recovery_dump(inputs, targets, scores, dataset,args):
    sorted_tuples = sorted([(input, target, score) for input, target, score in zip(inputs, targets, scores)], key= lambda x:x[2], reverse = True)
    with open(os.path.join(args.root, f'{dataset}_recovery_alpacasus.jsonl'), 'w') as write_file:
            for tuple in sorted_tuples:
                obj = {"input": tuple[0], "target": tuple[1], "score": tuple[2]}
                write_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'Running Recovery dump. Remaining sampling budget for dataset {args.budget - len(scores)}')

def invoke_gpt_scoring(messages, inputs, targets, dataset, args):
    responses, scores = [], []
    for message in tqdm.tqdm(messages): 
        try:
            response = client.chat.completions.create(model = "gpt-4o", messages = message, max_tokens = 3)
            sleep(0.005)
        except openai.AuthenticationError:
            print("Failed to authenticate. Check your API key.")
            recovery_dump(inputs[:len(scores)], targets[:len(scores)], scores, dataset, args)
        except openai.RateLimitError:
            print("Rate limit exceeded. Please try again later.")
            recovery_dump(inputs[:len(scores)], targets[:len(scores)], scores, dataset, args)
        except openai.APIError as e:
            print(e)
            recovery_dump(inputs[:len(scores)], targets[:len(scores)], scores, dataset, args)
        output = response.choices[0].message.content
        if output is None: 
            print('Returned None')
        score = extract_score(output)
        responses.append(output)
        scores.append(score)
    return responses, scores 

def make_review_messages(inputs, targets):
    system_prompt = '''We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following. \n\n'''
    user_prompt = '''Please rate the given instruction and response tuple according to the accuracy of the response. Each response receives a score on a scale of 0 to 5, where a higher score indicates higher level of the accuracy. Please first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.'''
    messages = []
    for input, target in zip(inputs, targets):
        eval_prompt = f'Instruction: {input} \n Response: {target}\n' + user_prompt
        message = [{"role":"system", "content": system_prompt},
                   {"role":"user", "content": eval_prompt},
                   ]
        messages.append(message)
    return messages 

def extract_score(response):
    if '0' in response:
        score = 0
    elif '1' in response:
        score = 1
    elif '2' in response:
        score = 2
    elif '3' in response:
        score = 3
    elif '4' in response:
        score = 4
    elif '5' in response:
        score = 5
    else: 
        score = -1
        print('Score not in acceptable range.')
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str, default='../sample_data/temp/', help='path to where full dataset is stored. This is also where the scored samples for the entire dataset will be stored. ')
    parser.add_argument('--budget',type=int, default=40000, help='the number of samples that are originally scored (typically a large subset (>50%) of the original dataset)')
    parser.add_argument('--prune', action='store_true', default=False, help='if scored samples need to be pruned.')
    parser.add_argument('--pruning_budget',type=int, default=10000, help='used for pruning the ranked samples according to the number of samples need (1000, 5000, 10000 for our experiment)')
    args = parser.parse_args()

    datasets = os.listdir(args.root)

    for dataset in datasets:
        if args.prune:
            try: 
                identifier = dataset.replace('_unpruned',f'_{args.pruning_budget}')
                with open(os.path.join(args.root, dataset), 'r') as read_file:
                    records = read_file.read().strip().split('\n')
                    print(len(records))
                    dump_objs = []
                    for record in records: 
                        if json.loads(record)['score'] >= 4: 
                            dump_objs.append({'input': json.loads(record)['input'], 'target': json.loads(record)['target'], 'score': json.loads(record)['score']})
                    print(f'{len(dump_objs)} number of eligible records!')
            
            
                with open(os.path.join(os.path.join(args.root,f'{identifier}')), 'w') as write_file: 
                        for obj in dump_objs[:args.pruning_budget]:
                            write_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except: 
                print(f'Failed for File: {dataset}')
        else:             
            print(f'Scoring {args.budget} random samples from {dataset}')
            # from random_long_sampler.py - the file name is {dataset}.json in the root folder
            with open(os.path.join(args.root, dataset), 'r') as file:
                records = file.read().strip().split('\n')
                print(f'Read record from {dataset}')

            inputs = [json.loads(record)['input'] for record in records]
            targets = [json.loads(record)['target'] for record in records]
        
            messages = make_review_messages(inputs, targets)[:args.budget]
            print(f'{len(messages)} samples will be scored.')
            start_time = time.time()
            explainations, scores = invoke_gpt_scoring(messages, inputs, targets, dataset, args)
            end_time = time.time()
            print(f'{(end_time -start_time)/60} minutes ...')
            sorted_tuples = sorted([(input, target, score) for input, target, score in zip(inputs, targets, scores)], key= lambda x:x[2], reverse = True)
            identifier = dataset.replace('.jsonl','')
            with open(os.path.join(args.root, f'{identifier}_alpacasus_unpruned.jsonl'), 'w') as write_file:
                for tuple in sorted_tuples:
                    obj = {"input": tuple[0], "target": tuple[1], "score": tuple[2]}
                    write_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        