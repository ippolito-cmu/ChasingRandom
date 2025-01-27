# This file is used for generating the reference files for comparing a strategy-specific selected data with the model trained on randomly subsampled data for each configuration
import os 
import json
import re
import random
import argparse
from utils import dump_alpacaeval_predictions

def parse_alpacaeval_outputs(file_path,args):
    with open(os.path.join(args.inference_root,file_path),'r') as file: 
        records = json.load(file)
    model_identifier = records[0]['generator']
    return model_identifier, [record['instruction'] for record in records], [record['output'] for record in records]


def dump_alpaceval_run_command(leaderboard_path, strategy_file, random_reference_file, script):
    command = f"""alpaca_eval make_leaderboard --leaderboard_path {leaderboard_path} --all_model_outputs {strategy_file} --reference_outputs {random_reference_file}.json --annotators_config alpaca_eval_gpt4_turbo_fn"""
    with open(script,'a') as file:
        file.write(command + '\n') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_root', type=str, default='../results', help='the folder where you have stored all alpacaeval inferences')
    parser.add_argument('--reference_root', type=str, default='../results/alpacaeval_ref/', help='the folder where you want to store all the reference_outputs')
    parser.add_argument('--leaderboard_path', type=str, default='../results/alpacaeval_leaderboard/', help='the folder where alpacaeval leaderboards will be dumped.')
    parser.add_argument('--script', type=str, default='../scripts/inference.sh', help='bash script which will log all the commands which you can run')
    args = parser.parse_args()
    files = os.listdir(args.inference_root)

    leaderboard_objects, reference_leaderboard_objects = [],[]
    if not os.exists(args.reference_root):
        os.mkdir(args.reference_root)
    if not os.exists(args.leaderboard_path):
        os.mkdir(args.leaderboard_path)


    for file in files: 
        if not os.path.isdir(file) and 'alpacaeval' in file:
            if 'full' in file:
                dataset, strategy, budget = file.split('_')[1:4]
                budget = budget.replace('.json','')
                for candidate_budget in ['1000','5000','10000']: 
                    if strategy == 'full':
                        budget = candidate_budget # We match the finetuned model with 10000 samples with the model trained using 10000 random samples
                    
                    
                    if strategy !='random' and strategy != 'strictrandom': #These are much fewer instances 
                        corresponding_random_outputs = {}
                        configuration = rf'.*_{dataset}_random_{budget}_.*'
                        
                        # Searching for all the files which have the same config 
                        corresponding_files = [random_file for random_file in files if re.match(configuration,random_file)]
                        print(f'For {dataset}, {strategy} and {budget} - Sampling responses from a set of {len(corresponding_files)} random-model responses ...')
                        if len(corresponding_files) < 3: 
                            print(f'WARNING! Reference outputs underflow for file {file}')
                        for corresponding_random_file in corresponding_files:
                            model_identifier, prompts, model_outputs = parse_alpacaeval_outputs(corresponding_random_file,args)
                            for prompt, model_output in zip(prompts, model_outputs):
                                if prompt in corresponding_random_outputs:
                                    corresponding_random_outputs[prompt].append(model_output)
                                else: 
                                    corresponding_random_outputs[prompt] = [model_output]

                        # Created the pool of all possible responses for the model - now sampling from these to create the corresponding random sample. 
                        print('Creating a pool of random-model responses as reference model outputs ...')
                        sampled_random_responses = [random.sample(responses,1)[0] for responses in corresponding_random_outputs.values()]
                        base_model_identifier = f'{dataset}_{strategy}_{budget}'

                
                        dump_alpacaeval_predictions(list(corresponding_random_outputs.keys()),
                                                        sampled_random_responses, 
                                                        os.path.join(args.reference_root, f'reference_outputs_for_{base_model_identifier}'),
                                                        f'reference_{base_model_identifier}'
                                                        )
                            
                        # also writing the python command for the main computation 
                        command = dump_alpaceval_run_command(os.path.join(args.leaderboard_path, f'{base_model_identifier}.csv'),
                                                os.path.join(args.inference_root, file), 
                                                os.path.join(args.reference_root, f'reference_outputs_for_{base_model_identifier}'),
                                                args.script)
                        
                        print(command)

        




