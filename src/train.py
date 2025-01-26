import wandb
from datasets import load_dataset
from dataclasses import dataclass, field
from tqdm import tqdm 
import torch
import random
from transformers import (    
    HfArgumentParser,         
    AutoConfig,
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
from trl import SFTTrainer
from transformers.integrations import WandbCallback
from accelerate import PartialState

@dataclass 
class ModelArguments:
    model_type: str = field(default=None)
    model_name: str = field(default = None)
    cache_dir: str = field(default=None)
    budget: int = field(default=9999999999999)
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    test_file: str = field(default=None)
    max_source_length: str = field(default=512)
    max_target_length: str = field(default=512)
    max_train_samples: int = field(default=None) 
    max_eval_samples: int = field(default=15)
    wandb_artefact_logging: bool = field(default=False)
    wandb_run_name: str = field(default=None)
    output_prediction_file: str = field(default='predictions.txt')
    previous_checkpoint: str = field(default=None)
    wandb_project_name: str = field(default=None)
    wandb_entity: str = field(default='hdiddee')

SYSTEM_INSTRUCTION = 'Below is an instruction that is optionally paired with some additional context. Respond appropriately follows using the context (if any) \n'

def format_test_instruction(prompt):
        # This does not account for template as its only used for sanity checking generations on wandb 
        return f"""{SYSTEM_INSTRUCTION} \n {prompt} \n """

class ComputeCallback(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=128, log_model="checkpoint"): 
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.shuffle(seed=42)
        self.sample_dataset = self.sample_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.max_new_tokens = max_new_tokens
        self.gen_config = AutoConfig.from_pretrained(trainer.model.name_or_path)

    def generate(self, prompt, max_new_tokens):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')
        with torch.inference_mode():
            output = self.model.generate(input_ids = tokenized_prompt.input_ids.cuda(), attention_mask = tokenized_prompt.attention_mask.cuda(), max_new_tokens= max_new_tokens, pad_token_id = self.tokenizer.eos_token_id)
            predictions = self.tokenizer.decode(output[0][len(prompt[0]):], skip_special_tokens=True)
        return predictions 
    
    def samples_table(self, examples, max_new_tokens):
        records_table = wandb.Table(columns=["prompt", "generation"])
        for example in tqdm(examples, leave=False):
            prompt = example["conversations"][0]
            prompt = format_test_instruction(prompt)
            generation = self.generate(prompt=prompt, max_new_tokens = max_new_tokens)
            records_table.add_data(prompt, generation)
        return records_table
    
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset, self.max_new_tokens)
        self._wandb.log({"sample_predictions":records_table})

    
def main():

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()    
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name = args.wandb_run_name, group=f'{args.wandb_run_name}_ddp')
    device_string = PartialState().process_index
    print(f'Device Map: {device_string}')
        
    if args.model_type == 'llama' or args.model_type=='llama2':
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                     attn_implementation="flash_attention_2", 
                                                     device_map = {'':device_string}, 
                                                     torch_dtype=torch.bfloat16)
        model.config.use_cache = False
        model.config.pretraining_tp = 1   
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    elif args.model_type == 'gemma':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                     attn_implementation="flash_attention_2",  
                                                     device_map = {'':device_string}, 
                                                     torch_dtype=torch.bfloat16)
        tokenizer.eos_token = '</s>'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"


    data_files = {}
    if args.train_file is not None: 
        data_files["train"] = args.train_file
    if args.validation_file is not None: 
        data_files["validation"] = args.validation_file
    if args.test_file is not None: 
        data_files["test"] = args.test_file
    raw_datasets = load_dataset('json', data_files=data_files, cache_dir=args.cache_dir)
    train_dataset = raw_datasets["train"]

    args.max_train_samples = min(args.budget, len(train_dataset))  # Setting max samples to training data budget 
    print(f'Budget set to: {args.max_train_samples}')
    try: 
        test_dataset = raw_datasets["test"]
    except:  
        print('No Test Set Found. Will be using LIMA.')
        test_dataset = load_dataset('GAIR/lima')['test']
    
    def format_instruction(model_input):
        if args.model_type == 'gemma': 
            return f"### USER: {SYSTEM_INSTRUCTION} \n ### Instruction: {model_input['inputs']}\n### ASSISTANT: {model_input['labels']}"
        if args.model_type == 'llama':
            return f"""{SYSTEM_INSTRUCTION} \n ### Instruction: {model_input['inputs']} \n ### Response: {model_input['labels']}"""
        if args.model_type == 'llama2':
            return [f"""<s> [INST] <<SYS>> {SYSTEM_INSTRUCTION} <</SYS>>{model_input} [/INST]"""]
        
    def preprocess_train_function(examples):         
        model_inputs = {}
        try: 
            inputs = [ex['input'] for ex in examples['task']]  # For FLAN-type datasets where the task-split was necessary to assert uniform sampling per task.  
            targets = [ex['target'] for ex in examples['task']]
        except:
            inputs = [ex for ex in examples['input']]
            targets = [ex for ex in examples['target']]

        model_inputs['inputs'] = inputs
        model_inputs['labels'] = targets 
        return model_inputs
    
    def preprocess_test_function(examples):         
        model_inputs = {}
        test_inputs = [ex[0] for ex in examples['conversations']]
        model_inputs['inputs'] = test_inputs
        return model_inputs
    
    if training_args.do_train:
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if 'random' in training_args.output_dir: # To maintain strict randomness in the sampling. 
            generated_seed = random.choice(range(args.max_train_samples))
            train_dataset = train_dataset.shuffle(seed=generated_seed)
            print(f'Have reshuffled dataset for random sampling with {generated_seed}...')
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
        preprocess_train_function,
        batched=True,
        desc=f"Creating input and target training samples with budget {args.max_train_samples}",
        )

    if training_args.do_eval:
        if args.max_eval_samples is not None:
            eval_dataset = train_dataset.select(range(args.max_eval_samples))
            eval_dataset = eval_dataset.map(
            preprocess_train_function,
            batched=True,
            desc="Creating input and target validation samples using the Training Dataset!.",
            )
            test_dataset = test_dataset.select(range(args.max_eval_samples))
            test_dataset = test_dataset.map(
                preprocess_test_function,
                batched=True,
                desc="Sampling LIMA Test Prompts for Checking Inference!",
            )


    set_seed(training_args.seed)
    training_args.ddp_find_unused_parameters = False

    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        max_seq_length=args.max_source_length, 
        train_dataset=train_dataset,
        eval_dataset= eval_dataset if training_args.do_eval else None, 
        packing = True, 
        tokenizer=tokenizer,
        formatting_func = format_instruction
    )
    
    if args.wandb_artefact_logging: 
        callback = ComputeCallback(trainer, test_dataset, num_samples=args.max_eval_samples, max_new_tokens=256) #Reduced max target length since we are only sanity checking 
        trainer.add_callback(callback)

    if args.previous_checkpoint is not None: 
        print('Found a previous checkpoint - so resuming training from there ...')
        train_result = trainer.train(resume_from_checkpoint = args.previous_checkpoint)
    else: 
        train_result = trainer.train()
    trainer.save_model()  
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    wandb.finish()


if __name__ == "__main__":
    main()
