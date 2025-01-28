### Introduction

Official Code for [Chasing Random: Instruction Selection Strategies Fail To Generalize](https://arxiv.org/abs/2410.15225). 

# Table of Contents
- [Environment](#environment)
- [Data Format and Selected Subsets](#data)
- [Finetuning Instructions](#fine-tuning)
- [Selection Strategies](#selection-strategies)
- [Evaluation](#evaluation)
  - [With vLLM](#with-vllm)
  - [Without vLLM](#without-vllm)
  - [Benchmark Specific Details](#Benchmark-Specific-Details)
- [Citations](#citations)

### Environment
requirements.txt contains the finetuning and data selection strategy installation setup. vllm_requirements.txt contains the vLLM installation requisites. OpenAI Key's has to be set for AlpacaEval's computation. 
```
pip install -r requirements.txt
pip install -r vllm_requirements.txt
export OPENAI_API_KEY=<YourKeyHere>
```

### Data 
- You can find all the data (including specific random,strictrandom run data for reproducibility and overlap assessment) in the data.zip hosted [here](https://storage.cloud.google.com/chasing_random/data.zip). 
- For efficiency, we do not include the full dataset for [EVOL](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k), [ALPACA](https://huggingface.co/datasets/tatsu-lab/alpaca) and [DOLLY](https://huggingface.co/datasets/databricks/databricks-dolly-15k) as we source them from huggingface. 
- We include our uniformly subsampled FLAN (our version of the full dataset) for reproducibility in this folder. To setup the environment for FLAN, use instructions provided [here](https://github.com/google-research/FLAN/). Then run the __seqio_creator.py__ (for example, with a different per task weightage). We recommend exporting the following variables to avoid exceeding Disk Quota and cloud authentication errors.  
```
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export TFDS_DATA_DIR=<REASONABLY LARGE PERSISTENT STORAGE>
```

### Data Format 
The main finetuning script train.py excepts the data to be formatted in the following jsonl format. You can find a sample data file in src/sample_data/. 
```
{"input":<instruction (including any input context)>, "target":<target>}
```

### Fine-Tuning
Run accelerate config setting with the available number of available GPUs. Edit the __src/scripts/train.sh__ --num_processes parameter to the same value. The hyperparameters in train.sh correspond to a 8-GPU setup so edit the hyperparameters accordingly if --num_processess != 8. 
```
accelerate config 
bash train.sh 
```


### Selection Strategies 
- *Alpacasus*: Run sampler.py to create initial pool of samples to be scored. Then run scorer.py to score the pool of samples; Finally run the scorer.py with pruning_budget to prune according to the subsampled budget. 
```
python scorer.py --budget 5 
python scorer.py --prune --pruning_budget 10
```
```Files of interest: sampler.py, scorer.py```

- *Longest*: Run __sampler.py__ and pass __--length_sorted__.
```
python sampler.py --root ../sample_data/temp/ --dolly --length_sorted --budget 10 --identifier temp 
```
```Files of interest: sampler.py, scorer.py```


- *Cherry*: We use the code opensourced by the authors listed [here](https://github.com/tianyi-lab/Cherry_LLM/blob/main/cherry_seletion/data_analysis.py).  You can use __cherry_data_converter.py__ to convert our data into the format required by the CherryLM training scripts (You will have to supply the path to your preprocessed FLAN dataset as we load that locally. Set this to flan_full.jsonl from data.zip if you're not using a custom FLAN dataset). Use __cherry.sh__ to run all the steps (training the pre-experienced model, clustering, scoring the samples for IFD and finally selecting samples.)
```
python cherry_data_converter.py --write_root ../sample_data/temp/
cd ..
git clone https://github.com/tianyi-lab/Cherry_LLM.git
cd scripts
bash cherry.sh
```
```Files of interest: cherry_data_converter.py, cherry.sh```

- *DEITA*: We directly use the code opensourced by the authors listed [here](https://github.com/hkust-nlp/deita); To convert data into the sharegpt format you can use __preprocessor/data_data_converter.py__. This file has 2 flags:
- --preprocessing: This is used for converting the original datasets into shareGPT format 
- --training: This is used for filtering the deita-scored data in accordance with whatever your sampling budget may be (default is 10000). 

This file will also give you a warning if your deita-scored samples are not enough to subsample for the budget of your choice. If you encounter an underflow, we recommend trying a lower value of __--threshold__ in __deita_selection.py__ (default for us is 0.9). The __deita_selection.py__ files shows sample usage of the entire scoring pipeline (embedding generation and scoring).

```
python deita_data_converter.py --preprocessing \
 --max_budget 100 --root ../sample_data/  \
 --write_root ../sample_data/temp \

git clone https://github.com/hkust-nlp/deita.git
cd deita
pip install -e .

python deita_selection.py --threshold 0.9

```
```Files of interest: deita_selection.py, deita_data_converter.py```

### Evaluation 
inference.py loads all the evaluation benchmarks used in the paper. By default - we use VLLM for faster inference and you can choose whichever variant based on the description below: 
- *With VLLM*:  We provide a separate __vllm_requirements.txt__ should you choose to run inference using VLLM. The current requirements.txt includes its installation and you can see __inference.sh__ for example usage.

- *Without VLLM*: If you prefer not using VLLM - you can refer to the __batch_process()__ function in the utils to run inference while only leveraging accelerate. For Eval Harness evaluations, you can refer to __openllm_simple_evaluate.py__ which leverages 4-bit quantized model inferencing with simple_evaluate for Eval Harness evaluations.

- *Benchmark Specific Details* 
  - *AlpacaEval*: You can use /evaluation/alpacaeval.py to create the random-baselines for AlpacaEval computation. This will also write the exact command to run for generating the alpacaeval leaderboards for each comparison. 
  - *IFEval*: You can use /evaluation/ifeval.py to compute IFEval scores using the inferences generated for ifeval. 
  - *Eval Harness*: You can use eval_harness_summary_compiler() in ./utils.py to parse all the results from Eval Harness inference into csv's for analysis. 
```
[Instruction Following Benchmarks] bash inference.sh 

[Eval Harness Evaluation] accelerate launch -m lm_eval --model hf --model_args pretrained=/data/user_data/hdiddee/llama_models/llama_checkpoint/,max_length=2048,dtype=auto --tasks arc_challenge,hellaswag,arc_easy,truthfulqa_mc1,truthfulqa_mc2,winogrande,mmlu  --batch_size auto --output_path ../results
```

### Citations 
- If you found our code and/or paper useful, please consider citing our work
```
@misc{diddee2024chasingrandominstructionselection,
      title={Chasing Random: Instruction Selection Strategies Fail to Generalize}, 
      author={Harshita Diddee and Daphne Ippolito},
      year={2024},
      eprint={2410.15225},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.15225}, 
}
```
- Code for cherry data selection was sourced from the official open source repository. 
```
@inproceedings{li-etal-2024-quantity,
    title = "From Quantity to Quality: Boosting {LLM} Performance with Self-Guided Data Selection for Instruction Tuning",
    author = "Li, Ming  and
      Zhang, Yong  and
      Li, Zhitao  and
      Chen, Jiuhai  and
      Chen, Lichang  and
      Cheng, Ning  and
      Wang, Jianzong  and
      Zhou, Tianyi  and
      Xiao, Jing",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.421",
    pages = "7595--7628",
}
```
- Code for deita data selection was sourced from the official open source repository. 
```
@inproceedings{
liu2024what,
title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning},
author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=BTKAeLqLMw}
}
```