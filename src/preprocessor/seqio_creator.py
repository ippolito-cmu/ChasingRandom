import seqio
from random import randint
import argparse 
import re 
import tensorflow as tf
from FLAN.flan.v2 import (
    task_configs, 
    templates, 
    utils, 
    constants, 
    preprocessors
    )

import json
import functools
import logging 
import os
from constants import INPUT_SEQ_LEN, TARGET_SEQ_LEN, QUAL_SAMPLES, template_type

def qual_eval_dump(dump, root):
    with open(f'{root}/qualitative_eval.jsonl', 'a') as batch_file: 
        for item in dump[:QUAL_SAMPLES]:
            batch_file.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info('Qualitative eval samples dumped!')

def de_templatize_instruction(expected_pattern):
    pattern = r"\{.*?\}"
    clean_text = re.sub(pattern, "", expected_pattern[0][0]) # Only separating the input and the instruction
    print(clean_text)
    instruction = clean_text
    return instruction

def dump_data(task: str, 
              root: str,
              expected_pattern: str,
              inputs: list[str],
              labels: list[str],
              identifier: str):
    
    original_identifier = identifier 
    dump = []
    for src, tgt in zip(inputs, labels):
        instruction = de_templatize_instruction(expected_pattern)
        item = {'instruction': instruction, 'input': src.replace(instruction,' '), 'target': tgt}
        dump.append(item)
        
    
    identifier = f'{root}/{identifier}.jsonl'       
    
    with open(identifier, 'a') as f:
        for item in dump:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_data(task: str,
                dataset: tf.data.Dataset,
                max_samples: int, 
                root: str, 
                identifier: str,
                expected_pattern: str): 
        inputs, labels = [], []
        
        for sample in dataset.take(max_samples).cache():
            inputs.append(sample['inputs_pretokenized'].numpy().decode('utf-8'))
            labels.append(sample['targets_pretokenized'].numpy().decode('utf-8'))
        
        dump_data(task, root, expected_pattern, inputs, labels, identifier)    


def register_task(task : str,
                  config : task_configs.TaskConfig, 
                  patterns: list[tuple[str, str, str]],
                  template_type: str, 
                  max_samples: int,
                  root: str, 
                  split: str, 
                  expected_pattern):
        add_template_metadata_fn = functools.partial(preprocessors.add_template_info, template_type=template_type)
        formatter = preprocessors.get_batch_formatter(patterns)
        task_status = -1 
            
        for suffix, output_features in constants.TRAIN_TASK_SUFFIXES_AND_FEATURES:
            seqio.TaskRegistry.add(
                task,
                source=config.source,
                preprocessors=config.preprocessors + [add_template_metadata_fn] + formatter + preprocessors.FLAN_TOKENIZE, 
                postprocess_fn=config.postprocess_fn,
                output_features=output_features,
                metric_fns=config.metric_fns)
        
        if task == 'unified_qa_science_inst': 
            split = 'train' # According to https://github.com/google-research/FLAN/blob/e9e4ec6e2701182c7a91af176f705310da541277/flan/v2/task_configs.py#L112
        try: 
            dataset = seqio.get_mixture_or_task(task).get_dataset(
            sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
            num_epochs=1,  
            shuffle=False, 
            split=split)
            try: 
                logging.info(f'{len(dataset)} number of records being used for {task}.')
            except TypeError: 
                logging.info(f'Length of dataset is unknown.')
        except Exception as e: 
            print(f'{task} could not be loaded because of {e}.')            
            return task_status  

        create_data(task, dataset, max_samples, root, identifier=split, expected_pattern=expected_pattern) 
        task_status = 1
        return task_status 

def registers_task_mixture(task_pool: str, 
                           max_samples: int, 
                           root: str, 
                           identifier: str):
    current_task_count, current_task_pool = 0, []
    for t_name, config in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items():
        if t_name in task_pool:
            print(f'Routing data for {t_name}')
            if 't0' in t_name: 
                print('Skipping since this has a buggy cache..')
                continue
            flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
            patterns_list = templates.PATTERNS[flan_pattern_name]
            try: 
                selected_patterns = [patterns_list[randint(0, len(patterns_list))]]
            except: 
                selected_patterns = patterns_list[0:1]
            print(f'Expected pattern: {selected_patterns}')
            expected_pattern = selected_patterns  # expected_pattern is the instruction
            logging_name = t_name
            task_status = register_task(t_name, config, selected_patterns,template_type, max_samples, root, identifier, expected_pattern) #Adding expected pattern for Cherry_LM
            
            if task_status != 1:
                logging.info(f'Failed for {t_name}')
            current_task_count += 1 
            current_task_pool.append(logging_name) 
            
    return current_task_count, current_task_pool


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_of_tasks', type = int, default = 280, help = 'Pass the total number of tasks you want to evaluate on (will mandatorily include held-in tasks)')
    parser.add_argument('--max_samples', type = int, default = 1000, help = 'Max number of examples to be sampled per task')
    parser.add_argument('--log_name', type = str, help = 'Log name')
    parser.add_argument('--root', type = str, default = 'temp', help = 'Training and Evaluation data path')

    args = parser.parse_args()
    if args.log_name:
        log_name = args.log_name 
    else: 
        log_name = args.root.split('/')[-1]
    logging.basicConfig(filename=f'./{log_name}.txt', encoding = 'utf-8', level = logging.INFO)
    TASK_POOL = list(task for task, _ in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items())
    print(F'Task pool has {len(TASK_POOL)} number of tasks.')

    if os.path.exists(args.root):
        logging.info('Path already exists. Will be appending to file.')
    else: 
        logging.info('Creating root path.')
        os.mkdir(args.root)

    # GOLD DATA CONSTRUCTION 
    logging.info('Begining Gold Data Construction!')
    RANDOMIZE = False # To reinitialize the state of the identifier (Used for creating the Gold Dataset)
    CURRENT_TASK_POOL = list(task for task, _ in task_configs.ALL_CANDIDATE_TASK_CONFIGS.items())
    CURRENT_TASK_COUNT, CURRENT_TASK_POOL = registers_task_mixture(CURRENT_TASK_POOL, args.max_samples, args.root, identifier= 'train') 
    logging.info(f'{CURRENT_TASK_COUNT} gold task datasets have been created and the task list is {CURRENT_TASK_POOL}.')
  