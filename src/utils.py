import evaluate
import numpy as np
from datasets import load_dataset 
import torch
import tqdm
import os 
import csv
import json
import random

IFFEVAL_DATA_DIR = "../instruction_following_eval/data/input_data.jsonl"
SYSTEM_INSTRUCTION = 'Below is an instruction that is optionally paired with some additional context. Respond appropriately follows using the context (if any) \n'

def model_formatter(prompts, model_identifier, system_instruction=SYSTEM_INSTRUCTION):
    if 'gemma' in model_identifier:
        return [f"{system_instruction} \n ### USER: {prompt}\n ### ASSISTANT:" for prompt in prompts]
    elif 'llama' in model_identifier:
        return [f"{system_instruction} \n ### Instruction: {prompt} \n ### Response:" for prompt in prompts]
    elif 'llama2' in model_identifier and 'llama' not in model_identifier:
        return [f"""<s> [INST] <<SYS>> {SYSTEM_INSTRUCTION} <</SYS>>{prompt} [/INST]""" for prompt in prompts]
    else: 
        print('WARNING: Model identifier did not have parent identifier. Defaulting to LLaMa')
        return [f"{system_instruction} \n ### Instruction: {prompt} \n ### Response:" for prompt in prompts]


def openllm_summary_compiler(inference_root='../results', compiled_result_root='../compiled_results'):
    '''Reads the Eval Harness outputs (json outputs generated through inference) and compiles them into readable csvs for analysis and visualization.'''
    if not os.exists(compiled_result_root):
        os.mkdir(compiled_results)

        files = os.listdir(inference_root)
        compiled_results = []
        for file in files:
            parts = file.split('_')
            type, perturbation = parts[0], parts[1].replace('.json','')        
            with open(os.path.join(inference_root,file),'r') as f:
                records = json.load(f)
                obj = ['openllm', type, perturbation, 
                    round(records['results']['mmlu']['acc,none'],3), 
                    round(records['results']['arc_easy']['acc,none'],3), 
                    round(records['results']['arc_challenge']['acc,none'],3),
                    round(records['results']['hellaswag']['acc,none'],3),
                    round(records['results']['truthfulqa_mc1']['acc,none'],3),
                    round(records['results']['truthfulqa_mc2']['acc,none'],3),
                    round(records['results']['winogrande']['acc,none'],3),
                    ]
                compiled_results.append(obj)
    
        with open(os.path.join(compiled_result_root,'openllm.csv'),'a') as file:
            writer = csv.writer(file)
            writer.writerow(['Benchmark','Dataset','Strategy', 'Budget', 'MMLU','ARC-e','ARC-c','HellaSwag','TruthfulQA_1', 'TruthfuQA_2','Winogrande'])
            for obj in compiled_results:
                writer.writerow(obj)

def calculate_accuracy(predictions, labels):
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy['accuracy']


def check_repeat(identifier, output_dir, benchmark):
    '''To check if inference is not being repeated accidentally.'''
    files = os.listdir(output_dir)
    file_identifier = f'{identifier}.jsonl'
    if file_identifier in files or f'{benchmark}_{identifier}.jsonl' in files:
        return -1
    return 0
    
def dump_infobench_predictions(dataset, predictions, identifier):
    with open(f'{identifier}.json', 'w') as write_file:
        for dataset_item, prediction in zip(dataset, predictions):
            dataset_item.update({'output': prediction})
            write_file.write(json.dumps(dataset_item, ensure_ascii=False))
            write_file.write('\n')

def batch_process(batch_size, model, tokenizer, prompts, max_new_tokens):
    """Vanilla batch inferencing (without VLLM) in case VLLM is not supported for inference. """
    batch_ctr = 0
    outputs = []
    failed_input = 0
    pbar = tqdm.tqdm(total=len(prompts))
    while batch_ctr < len(prompts):
        local_batch_prompt = prompts[batch_ctr:batch_ctr + batch_size]
        local_batch = tokenizer(local_batch_prompt, return_tensors = 'pt', truncation=True, padding='max_length', max_length = 512)# Has to be enough to include the prompt as well as the response. 
        with torch.no_grad():
            intermediate_outputs = model.generate(input_ids = local_batch.input_ids.cuda(), attention_mask = local_batch.attention_mask.cuda(),  max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id)
            predictions = tokenizer.batch_decode(intermediate_outputs[:, local_batch.input_ids.shape[-1]:], skip_special_tokens=True)
            outputs.append(predictions)
        batch_ctr += batch_size
        pbar.update()
    pbar.close()
    print(f'Failed for {failed_input} samples')
    return outputs




def set_max_token_length(labels):
    """Useful for Non-VLLM inference where max sequence length per benchmark may need to be adjusted per dataset (and using an arbitary high value may not be efficient.)"""
    sample_lengths = [len(label.split(' ')) for label in labels[:20]]
    if np.mean(sample_lengths) <= 8:
        return 8
    elif np.mean(sample_lengths) <= 64:
        return 64 
    elif np.mean(sample_lengths) <= 512:
        return 512 
    return 1024 

def read_iffeval_prompts(input_json_path):
    """
    Reading the input prompts for the iffeval dataset. 
    """
    with open(input_json_path, 'r') as file: 
        records = file.read().strip().split('\n')
    input_prompts = []
    for record in records: 
        input_prompts.append(json.loads(record)["prompt"])
    return input_prompts 

def dump_iffeval_predictions(prompts, model_responses, path):
    """Dumps the evaluating model's inferences in the required format specified here https://github.com/google-research/google-research/tree/master/instruction_following_eval"""
    dump = []
    for prompt, model_response in zip(prompts, model_responses):
        dump.append({"prompt": prompt, "response": model_response})
    with open(path + '.jsonl', 'a') as f:
        for item in dump:
            f.write(json.dumps(item, ensure_ascii=False) + '\n') 


def dump_alpacaeval_predictions(prompts, model_responses, path, model_identifier):
    """Dumps the evaluating model's inferences in the required format specified here https://github.com/tatsu-lab/alpaca_eval.
    This is also our default output format for other benchmarks except IFEval (which has a custom format)"""
    dump = []
    dataset = model_identifier.split('_')[0]
    for prompt, model_response in zip(prompts, model_responses):
        dump.append({"instruction": prompt, "output": model_response, "generator": model_identifier, "dataset":dataset, "datasplit":'test'})
    
    with open(path + '.json', 'w') as file:
        json.dump(dump, file, indent=4) 


def read_task_prompts(path):
    "Used for reading FLAN type prompts (which have a task-type metadata header)"
    with open(path, 'r') as file: 
        records = file.read().strip().split('\n')
    prompts = [json.loads(record)['task']['input'] for record in records]
    labels = [json.loads(record)['task']['target'] for record in records]
    return prompts, labels 
