import subprocess
import argparse
import csv
import re
import os 



def execute_command(input_file):
    command =  f'''python3 -m instruction_following_eval.evaluation_main --input_data=./instruction_following_eval/data/input_data.jsonl --input_response_data={input_file} --output_dir=./'''
    log = []
    with subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) as proc:
        try:   
            for line in proc.stdout:
                if 'prompt-level' in line: 
                    print(line, end='')
                    pattern = r"prompt-level: (\-?\d+\.\d+)"
                    match = re.search(pattern, line)
                    log.append(round(float(match.group(1)),4))
                if 'instruction-level' in line:
                    print(line, end='')
                    pattern = r"instruction-level: (\-?\d+\.\d+)"
                    match = re.search(pattern, line)
                    log.append(round(float(match.group(1)),4))
                return input_file, log[0], log[1]
        except: 
            print(f'Failed for {input_file}')

def inference(files, path_prefix, args):
    for file in files: 
            if 'iffeval' in file:
                print(f'Running for {file} file')         
                system, prompt_perf, instruction_perf = execute_command(os.path.join(path_prefix, file))
                
                with open(os.path.join(args.write_root, args.summary_file), 'a') as summary_file:
                        writer = csv.writer(summary_file)
                        writer.writerow(['iffeval',file, prompt_perf, instruction_perf]) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='path to where all inferences are stored.')
    parser.add_argument('--write_root', help='path to where the compiled results are to be stored')
    parser.add_argument('--summary_file', help='name of the compiled results file', default='ifeval_summary.csv')
    args = parser.parse_args()

    with open(os.path.join(args.write_root, args.summary_file), 'a') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Benchmark','Perturbation','prompt-level-acc','instruct-level-acc'])
        

    directories = os.listdir(args.root)
    retry = []
    for directory in directories: 
        if os.path.isdir(os.path.join(args.root, directory)):
            print(f'Running for {directory} directory')
            files = os.listdir(os.path.join(args.root, directory))
            path_prefix = os.path.join(args.root, directory)       
            inference(files, path_prefix, args)
        else:
            print(f'Will retry for {directory} as its not nested.')
            retry.append(directory)

    if len(retry) is not None: 
        inference(retry, args.root, args)
