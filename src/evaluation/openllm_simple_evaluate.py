import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import lm_eval 
from lm_eval.models.huggingface import HFLM
import argparse
import json
import os 
device  = 'cuda'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = '/scratch/')
    parser.add_argument('--model_type', type = str, default = 'llama2')
    parser.add_argument('--bs', type = int, default = 16)
    parser.add_argument('--lm_eval_harness_results_path', type = str, default = './')

    args = parser.parse_args()
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
    model = AutoModelForCausalLM.from_pretrained(
            args.model,
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            quantization_config = quant_config,   
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model,  
                                              trust_remote_code=True)
    lm = HFLM(pretrained=model, 
              tokenizer=tokenizer, 
              dtype=torch.bfloat16, 
              max_length=1024, 
              batch_size= args.bs, 
              trust_remote_code=True)
    
    results = lm_eval.simple_evaluate(lm, 
                                      tasks = ['arc_challenge','hellaswag', 'arc_easy', 'truthfulqa_mc1',  'truthfulqa_mc2', 'winogrande', 'mmlu'], 
                                      task_manager = lm_eval.tasks.TaskManager(),  
                                      num_fewshot=5)
    
    filtered_results = results.copy()  
    filtered_results = {key: value for key, value in results.items() if key != "samples"}  
    json_filtered_results = json.dumps(filtered_results, indent=4)  
    with open(os.path.join(args.lm_eval_harness_results_path, args.model_path), "w") as json_file:
        json_file.write(json_filtered_results)
