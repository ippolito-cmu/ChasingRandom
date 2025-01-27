# Sample command for generating inferences for instruction following benchmarks 
python ../inference.py --model_path /scratch/ \
--model_type llama \
--output ../results \
--benchmarks iffeval alpacaeval lima \

# Sample command for generating inferences for Eval Harness Benchmark 
python ../inference.py --model_path /scratch/ \
--model_type llama \
--output ../results \
--eval_harness \

# Sample command for creating AlpacaEval Leaderboards which compare strategy-tuned models with randomly sub-sampled data trained models 
python ../evaluation/alpacaeval.py \
--inference_root ../results \
--reference_root ../results/alpacaeval_ref/ \
--leaderboard_path ../results/alpacaeval_leaderboard/ \
--script ../scripts/test.sh