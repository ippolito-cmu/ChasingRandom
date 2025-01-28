import os
import json
from datasets import load_dataset 
from constants import DOLLY_HF_NAME, EVOL_HF_NAME, ALPACA_HF_NAME
import argparse
def dump_data(instructions, inputs, targets, dataset, path):
    dump = []
    if len(instructions) == 0:
        instructions = ['']*len(inputs) # For datasets like FLAN and EVOL which do not have the input-instruction distinction
    for instruction, input, output in zip(instructions, inputs, targets):
        dump.append({'instruction':instruction, 'input': input, 'output':output})
    with open(os.path.join(path, f'{dataset}.json'), 'w') as file:
        json.dump(dump, file)

def make_dolly(write_root):
    train_dataset = load_dataset(DOLLY_HF_NAME)['train'] 
    instructions = [ins for ins in train_dataset['instruction']]
    inputs = [cont for cont in train_dataset['context']]
    targets = [ex for ex in train_dataset['response']]
    dump_data(instructions, inputs, targets, 'dolly', write_root)

def make_evol_196k(write_root):
    train_dataset = load_dataset(EVOL_HF_NAME)['train'] 
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset[:100000] 
    inputs = [ex[0]['value'] for ex in train_dataset['conversations']]
    targets = [ex[1]['value'] for ex in train_dataset['conversations']]
    dump_data([], inputs, targets, 'evol', write_root)


def make_flan(write_root,
              FLAN_PROCESSED_DATA='../sample_data/temp.jsonl'):
    with open(FLAN_PROCESSED_DATA,'r') as file:
        records = file.read().strip().split('\n')
        inputs = [json.loads(record)['input'] for record in records]
        targets = [json.loads(record)['target'] for record in records]
    dump_data([], inputs, targets, 'flan', write_root)


def make_alpaca(write_root):
    train_dataset = load_dataset(ALPACA_HF_NAME)['train']

    instructions = [ex['instruction'] for ex in train_dataset]
    inputs = [ex['input'] for ex in train_dataset]
    targets = [ex['output'] for ex in train_dataset]
    dump_data(instructions, inputs, targets, 'alpaca', write_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_root', type=str, required=True)
    args = parser.parse_args()    
    make_dolly(args.write_root)
    make_alpaca(args.write_root)
    make_evol_196k(args.write_root)
    make_flan(args.write_root,
              FLAN_PROCESSED_DATA = '../sample_data') # path to FLAN-type data 

    