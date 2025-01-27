from vllm import LLM, SamplingParams 
from datasets import load_dataset
from utils import (model_wise_formatter,
                   dump_iffeval_predictions)
import torch
import os
import tqdm
import json
import argparse
import subprocess

def make_command(model, tasks, output_path):
    command = f'accelerate launch -m lm_eval --model hf --model_args pretrained={model},max_length=2048,dtype=auto --tasks {tasks}  --batch_size auto --output_path {output_path}'
    return command

def dump_predictions(prompts, model_responses, path, model_identifier):
    dump = []
    dataset = model_identifier.split('_')[0]
    for prompt, model_response in zip(prompts, model_responses):
        dump.append({"instruction": prompt, 
                     "output": model_response, 
                     "generator": model_identifier, 
                     "dataset":dataset, 
                     "datasplit":'test'})
    
    with open(path + '.json', 'w') as file:
        json.dump(dump, file, indent=4) 

def inference(test_inputs, model, sampling_params):
    with tqdm.tqdm(torch.no_grad()):
        intermediate_outputs =  model.generate(test_inputs, sampling_params)
        predictions = [intermediate_output.outputs[0].text for intermediate_output in intermediate_outputs]
    return predictions 

def load_model(args):    
    model = LLM(args.model_path)
    print('Loaded model with VLLM ...')
    sampling_params = SamplingParams(max_tokens = 1024, logprobs=None)
    return model, sampling_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, help='Required to format the prompts in accordance with the model class.')
    parser.add_argument("--output_path", type=str, default = '../results')
    parser.add_argument("--max_samples", type = int, default = None)
    parser.add_argument("--benchmarks", nargs='+', choices=['lima', 'alpacaeval','ifeval','infobench'],
                        help="List of benchmarks to run. Possible choices: ifeval, infobench, lima, alpacaeval")
    parser.add_argument("--eval_harness", action='store_true', default=False, help='if openLLM evaluations are to be conducted, needs to be passed')
    parser.add_argument("--tasks", nargs='+', 
                        default=['arc_challenge','hellaswag','arc_easy','truthfulqa_mc1','truthfulqa_mc2','winogrande','mmlu']
                        help='the openLLM tasks you want to run inference for')

    args = parser.parse_args()
    model_identifier = args.model_path.split('/')[-1]
    print(f'Model Identifier:{model_identifier}')

    if not args.eval_harness: 
        model, sampling_params = load_model(args)

        for benchmark in args.benchmarks:

            if  benchmark == 'infobench':        
                dataset = load_dataset("kqsong/InFoBench")['train']
                inputs = [record['input'] + ' ' + record['instruction'] for record in dataset]
            
            if benchmark == 'alpacaeval': 
                dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
                inputs = [record for record in dataset['instruction']]
            
            if benchmark == 'lima':
                dataset = load_dataset('GAIR/lima')['test']
                inputs = [record for record in dataset['conversations']]

                
            test_inputs = model_wise_formatter(inputs, args.model_type)
            outputs = inference(test_inputs, model, sampling_params)
            if not os.path.exists(os.path.join(args.output_path,benchmark)):
                os.makedirs(os.path.join(args.output_path,benchmark))
            
            dump_predictions(inputs, outputs, os.path.join(args.output_path,benchmark, model_identifier), args.model_path)

            if benchmark == 'ifeval':
                inputs = load_dataset('google/IFEval')['train']['prompt']            
                test_inputs = model_wise_formatter(inputs, args.model_type)
                outputs = inference(test_inputs, model, sampling_params)
                if not os.path.exists(os.path.join(args.output_path,benchmark)):
                    os.makedirs(os.path.join(args.output_path,benchmark))

                dump_iffeval_predictions(inputs, outputs, os.path.join(args.output_path,benchmark, model_identifier))

    else: 
        print('Running Inference for OpenLLM Benchmark')
        command =  make_command(args.model_path, ','.join(args.tasks), args.output_path)
        print(command)
        subprocess.run(command)

                



