### Environment
```
pip install -r requirements.txt
```
## [Optional]
- For alpacasus and AlpacaEval evaluation - OpenAI Key's has to be set. Set key using
``` export OPENAI_API_KEY=<YourKeyHere> ```


### Data 
You can find all the data (including specific random,strictrandom run data for reproducibility and overlap assessment) in the data.zip hosted [here](https://storage.cloud.google.com/chasing_random/data.zip). The data in this format is compatible with the finetuning format of the train.py script. For efficiency, we do not include the full dataset for [EVOL](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k), [ALPACA](https://huggingface.co/datasets/tatsu-lab/alpaca) and [DOLLY](https://huggingface.co/datasets/databricks/databricks-dolly-15k) as we source them from huggingface. We include our uniformly subsampled FLAN (our version of the full dataset) for reproducibility in this folder. To create your own FLAN dataset with a different per task weightage you can also use the __seqio_creator.py__. 

### Data Format 
The main finetuning script train.py excepts the data to be formatted in the following jsonl format. You can find a sample data file in src/sample_data/
```
{"input":<instruction (including any input context)>, "target":<target>}
```

### Fine-Tuning Instructions 
Run accelerate config setting with the available number of available GPUs. Edit the src/scripts/train.sh --num_processes parameter to the same value. The hyperparameters in train.sh correspond to a 8-GPU setup so edit the hyperparameters accordingly if --num_processess != 8. 
```
accelerate config 
bash train.sh 
```

### Inference
- **With VLLM*: By default - we use VLLM for faster inference. We provide a separate vllm_requirements.txt should you choose to run inference using VLLM. The current requirements.txt includes its installation and you can see inference.sh for example usage. 
- **Without VLLM*: If you prefer not using VLLM - you can refer to the __batch_process()__ function in the utils to run inference while only leveraging accelerate. For Eval Harness evaluations, you can refer to /additional/__openllm_simple_evaluate.py__ which leverages 4-bit quantized model inferencing with simple_evaluate for Eval Harness evaluations.
- This script uses inference.py for all the instruction following benchmarks (IFEVAL, ALPACAEVAL, LIMA) used in the paper. For openLLM evaluation we use the eval harness (which is also included in the install setup). 
You can pass the --eval_harness flag in inference.sh to run those evaluations with your tasks of choice. 
```
bash inference.sh 
```

### Selection Strategies 
- Alpacasus: Run sampler.py to create initial pool of samples to be scored. Then run scorer.py to score the pool of samples; Finally run the scorer.py with pruning_budget to prune according to the subsampled budget. 
```Files of interest: sampler.py, scorer.py```
- Longest: Run __sampler.py__ and pass __--length_sorted__
```Files of interest: sampler.py, scorer.py```
- Cherry: We use the code opensourced by the authors listed [here](https://github.com/tianyi-lab/Cherry_LLM/blob/main/cherry_seletion/data_analysis.py). Cherry LLM's data requires that the instruction and input (where input is the additional context on which the instruction is applied); FLAN and EVOL do not allow for this distinction but DOLLY, ALPACA do. Additionally, their training script requires daat to be formatted with different keys; You can use __cherry_data_converter.py__ . Following this you can use __cherry.sh__ to run all the steps (training the pre-experienced model, clustering, scoring the samples for IFD and finally selecting samples.)
```Files of interest: cherry_data_converter.py, cherry.sh```
- DEITA: We directly use the code opensourced by the authors listed [here](https://github.com/hkust-nlp/deita); Note that deita requires data to be formatted in the sharegpt format. To convert data into the sharegpt format you can use __preprocessor/data_data_converter.py__. Note that this file has 2 flags: --preprocessing: This is used for converting the original datasets into shareGPT format and --training: This is used for filtering the deita-scored data in accordance with whatever your sampling budget may be (default is 10000). This file will also give you a warning if your deita-scored samples are not enough to subsample for the budget of your choice. If you encounter an underflow, we recommend trying a lower value of __--threshold__ in __deita_selection.py__ (default for us is 0.9). The __deita_selection.py__ files shows sample usage of the entire scoring pipeline (embedding generation and scoring).
```Files of interest: deita_selection.py, deita_data_converter.py```



### Citations 
- Code for cherry data selection was sourced from the official open source repository. 

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

- code for deita data selection was sourced from the official open source repository. 

@inproceedings{
liu2024what,
title={What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning},
author={Wei Liu and Weihao Zeng and Keqing He and Yong Jiang and Junxian He},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=BTKAeLqLMw}
}