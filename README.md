### Environment
```
pip install -r requirements.txt
```
### Description


### Data 
You can find all the data (including specific random,strictrandom run data for reproducibility and overlap assessment) in the data.zip folder. Their format is compatible with the finetuning format of the train.py script. For efficiency, we do not include the full dataset for [EVOL](https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k), [ALPACA](https://huggingface.co/datasets/tatsu-lab/alpaca) and [DOLLY](https://huggingface.co/datasets/databricks/databricks-dolly-15k) as we source them from huggingface. We include our uniformly subsampled FLAN (our version of the full dataset) for reproducibility in this folder. 

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
