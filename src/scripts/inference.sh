# Sample command for generating inferences for instruction following benchmarks 
# python ../inference.py \
# --model_type llama \
# --output ../results \
# --benchmarks ifeval alpacaeval lima \

# Sample command for generating the command for Eval Harness Benchmark 
# python ../inference.py \
# --model_type llama \
# --output ../results \
# --eval_harness \

# accelerate launch -m lm_eval --model hf \
# --model_args pretrained=/data/user_data/hdiddee/llama_models/llama_checkpoint/,max_length=2048,dtype=auto \
# --tasks arc_challenge,hellaswag,arc_easy,truthfulqa_mc1,truthfulqa_mc2,winogrande,mmlu  \
# --batch_size auto \
# --output_path ../results


# # Sample command for creating AlpacaEval Leaderboards which compare strategy-tuned models with randomly sub-sampled data trained models 
python ../evaluation/alpacaeval.py \
--inference_root ../results \
--reference_root ../results/alpacaeval_ref/ \
--leaderboard_path ../results/alpacaeval_leaderboard/ \
--script ../scripts/test.sh