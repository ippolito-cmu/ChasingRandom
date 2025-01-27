from datasets import load_dataset 
import argparse
from constants import (LIMA_HF_NAME, 
                       TULU_HF_NAME, 
                       ALPACA_HF_NAME, 
                       EVOL_HF_NAME,
                       DOLLY_HF_NAME)
import json
skip_datasets = ['lima', 'wizardlm', 'gpt4_alpaca']

def make_flan(identifier, args):
    print('Caution: This is a pre-packed version of the dataset. Use the FLAN specific dataset_creator.py to create a non-packed version!!!!')
    if identifier is None: 
        identifier = 'flan'
    train_dataset = load_dataset(TULU_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed)
    train_dataset = train_dataset.filter(lambda example: example['dataset'] == 'flan_v2')
    
    print(f'After filtering {len(train_dataset)} samples remain.')
    if args.length_sorted: 
        if identifier is None:
            identifier = 'flan_longest'
        train_dataset = train_dataset.add_column('target_len',[len(example['messages'][1]['content']) for example in train_dataset])
        train_dataset = train_dataset.sort('target_len', reverse = True) #longest to short

                                                                    
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset))))
    inputs = [ex[0]['content'] for ex in train_dataset['messages']]
    targets = [ex[1]['content'] for ex in train_dataset['messages']]

    dump_samples(inputs, targets, args.root,identifier)

def make_tulu(identifier, args):
    if identifier is None: 
                identifier = 'tulu'
    train_dataset = load_dataset(TULU_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed) #Shuffling
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset)))) # Has not invoked skipping the datasets so can include LIMA and FLAN
    inputs = [ex[0]['content'] for ex in train_dataset['messages']]
    targets = [ex[1]['content'] for ex in train_dataset['messages']]
    dump_samples(inputs, targets, args.root,identifier)

def make_alpaca(identifier, args):
    if identifier is None: 
                identifier ='alpaca'
    train_dataset = load_dataset(ALPACA_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.length_sorted:  
        if identifier is None: 
            identifier = 'alpaca_longest'
        train_dataset = train_dataset.add_column('target_len',[len(example['output']) for example in train_dataset])
        train_dataset = train_dataset.sort('target_len', reverse=True)   
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset))))
    inputs = [ex['instruction'] + ex['input'] for ex in train_dataset]
    targets = [ex['output'] for ex in train_dataset]
    dump_samples(inputs, targets, args.root,identifier)

def make_evol_196k(identifier, args):
    if identifier is None:
            identifier = 'evol'
    train_dataset = load_dataset(EVOL_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.length_sorted:  
        if identifier is None: 
            identifier = 'evol_longest'
        train_dataset = train_dataset.add_column('target_len',[len(example[1]['value']) for example in train_dataset['conversations']])
        train_dataset = train_dataset.sort('target_len', reverse=True)   
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset))))
    inputs = [ex[0]['value'] for ex in train_dataset['conversations']]
    targets = [ex[1]['value'] for ex in train_dataset['conversations']]
    dump_samples(inputs, targets, args.root,identifier)
     
     
def make_evol_70k(identifier, args):
    if identifier is None:
            identifier = 'evol'
    train_dataset = load_dataset(EVOL_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed)
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset))))
    inputs = [ex['instruction'] for ex in train_dataset]
    targets = [ex['output'] for ex in train_dataset]
    dump_samples(inputs, targets, args.root,identifier)


def make_dolly(identifier, args):
    if identifier is None: 
                identifier = 'dolly'
    train_dataset = load_dataset(DOLLY_HF_NAME)['train']
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.length_sorted:  
        if identifier is None: 
            identifier = 'dolly_longest'
        train_dataset = train_dataset.add_column('target_len',[len(example) for example in train_dataset['response']])
        train_dataset = train_dataset.sort('target_len', reverse=True)   
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset)))) 
    inputs = [ins + cont for ins, cont in zip(train_dataset['instruction'], train_dataset['context'])]
    targets = [ex for ex in train_dataset['response']]
    dump_samples(inputs, targets, args.root,identifier)


def make_lima(identifier, args):
    if identifier is None: 
        identifier = 'lima'
    train_dataset = load_dataset(LIMA_HF_NAME)['train']
    train_dataset = train_dataset.select(range(min(args.budget, len(train_dataset)))) 
    if args.length_sorted:  
        if identifier is None:
            identifier = 'lima_longest'
        train_dataset = train_dataset.add_column('target_len',[len(example[1]) for example in train_dataset['conversations']])
        train_dataset = train_dataset.sort('target_len', reverse = True)
       
    inputs = [ex[0] for ex in train_dataset['conversations']]
    targets = [ex[1] for ex in train_dataset['conversations']]
    dump_samples(inputs, targets, args.root,identifier)


def dump_samples(inputs, targets, root, identifier):
    print(identifier)
    with open(f'{root}{identifier}.jsonl','a') as file:
        for input, target in zip(inputs, targets):
            obj = {"input":input, "target": target}
            file.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default='./')
    parser.add_argument('--lima', action='store_true', default=False)
    parser.add_argument('--flan', action='store_true',default=False)
    parser.add_argument('--dolly', action = 'store_true', default = False)
    parser.add_argument('--tulu', action='store_true',default=False)
    parser.add_argument('--alpaca', action='store_true',default=False)
    parser.add_argument('--evol_70k', action='store_true',default=False)
    parser.add_argument('--evol_196k', action='store_true', default=False)
    parser.add_argument('--length_sorted', action='store_true', default = False)
    parser.add_argument('--budget', type = int, default=99999999999999, help='Default picks the entire dataset.')
    parser.add_argument('--seed', type=int, default = 42)
    parser.add_argument('--identifier', type = str, default = None)
    parser.add_argument('--length_threshold', type = int, default=500)

    args = parser.parse_args()
    if args.identifier is not None: 
        identifier = args.identifier
        print(f'Running for budget {args.budget}')
    if args.lima:
        make_lima(identifier, args)
                    
    if args.flan:
        make_flan(identifier,args)

    if args.dolly: 
        make_dolly(identifier, args)

    if args.tulu: 
        make_tulu(identifier, args)
        
    if args.alpaca:
        make_alpaca(identifier, args)

    if args.evol_70k: 
        make_evol_70k(identifier, args)

    if args.evol_196k:
        make_evol_196k(identifier, args)