# Converts (a) datasets to the share-gpt format required for deita scoring and (b) filters the deita-scored data according to your budget
# https://huggingface.co/datasets/hkust-nlp/deita-6k-v0
import argparse
import os 
import random
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', action='store_true', default=False)
    parser.add_argument('--max_budget', default=100000, type=int, help='sets the maximum number of samples you want to sample from for training the deita embedding models. Typically len(dataset) for datasets <100K. ')
    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--root','--dataset_root', default = '../sample_data/', help='Path to where full datasets are stored.')
    parser.add_argument('--write_root', default='../sample_data', help='Wherever you want to store the final deita-filtered files.')
    parser.add_argument('--filter_budget', default=1000, type=int, help='The number of samples you want to filter from the deita-scored data.')
    args = parser.parse_args()
    
    if args.preprocessing:
        files = os.listdir(args.root)
        for file in files: 
                    try: 
                        dataset = file.split('_')[0]
                        print(f'Converting into Deita Format for {dataset}')
                        with open(os.path.join(args.root,file), 'r') as read_file: 
                            records = read_file.read().strip().split('\n')
                            try: 
                                human_instructions = [json.loads(record)['input'] for record in records]
                                bot_responses = [json.loads(record)['target'] for record in records]
                            except: 
                                human_instructions = [json.loads(record)['task']['input'] for record in records]
                                bot_responses = [json.loads(record)['task']['target'] for record in records]
                        converted_records = [[{'from': 'human', 'value': human}, 
                                                {'from':"gpt",'value':bot}] 
                                                for human, bot in zip(human_instructions, bot_responses)] #this assignment to "gpt" is only done for compatibility with the training pipeline
                    
                        sampled_converted_records = random.sample(converted_records, min(args.max_budget, len(converted_records)))
                        with open(os.path.join(args.root,f'deita_{dataset}.jsonl'),'w') as write_file:
                            for record in sampled_converted_records: 
                                write_file.write(json.dumps({'conversations':record}) + '\n')  
                    except: 
                      print(f'Failed for {file}')

    if args.training:
        files = os.listdir(args.root)
        for file in files: 
            dataset = file.split('_')[0]
            strategy = 'deita'
            with open(os.path.join(args.root,file), 'r') as read_file: 
                records = json.load(read_file)
                inputs = [record['conversations'][0]['value'] for record in records]
                targets = [record['conversations'][1]['value'] for record in records]
        
            if len(inputs) == len(targets) < args.filter_budget:
                print(f'Underflow for {dataset}! Only {len(inputs)} number of samples.')
            write_identifier = f'{dataset}_{strategy}_{args.filter_budget}.jsonl'
            with open(os.path.join(args.write_root, write_identifier),'w') as write_file:
                for input, target in zip(inputs, targets):
                    write_file.write(json.dumps({'input':input, 'target': target}) + '\n')
        