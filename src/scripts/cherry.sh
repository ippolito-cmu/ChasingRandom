
python ../Cherry_LLM/cherry_seletion/data_analysis.py \
    --data_path ../sample_data/lima_data.json \
    --save_path /scratch/temp_cherry_pre.pt \
    --model_name_or_path /data/user_data/hdiddee/llama_models/llama_checkpoint/ --prompt wiz \
    --max_length 512 \
    --mod pre

python ../Cherry_LLM/cherry_seletion/data_by_cluster.py \
    --pt_data_path /scratch/temp_cherry_pre.pt \
    --json_data_path ../sample_data/lima_data.json \
    --json_save_path ../sample_data/lima_full_pre.json \
    --sample_num 10 \
    --kmeans_num_clusters 10 \
    --low_th 25 \
    --up_th 75


# Code for training pre-experienced model: Can use the train.py for the same. Sample usage here. 

# accelerate launch --num_processes 8 ../train.py \
#                 --model_type llama \
#                 --model_name /data/user_data/hdiddee/llama_models/llama_checkpoint/ \
#                 --wandb_run_name $wandb_run_name \
#                 --wandb_project_name "test-run" \
#                 --do_train \
#                 --num_train_epochs 3 \
#                 --budget $budget \
#                 --train_file <cherry-clustered data: ../sample_data/lima_full_pre.json> \
#                 --save_strategy no \
#                 --learning_rate 2e-5 \
#                 --per_device_train_batch_size 2 \
#                 --gradient_accumulation_steps 8 \
#                 --optim paged_adamw_32bit \
#                 --weight_decay 0.0 \
#                 --fp16 False \
#                 --bf16 True \
#                 --save_total_limit 1 \
#                 --output_dir <set output directory path> \
#                 --warmup_ratio 0.03 \
#                 --lr_scheduler_type linear

python ../Cherry_LLM/cherry_seletion/data_analysis.py \
    --data_path ../sample_data/lima_data.json \
    --save_path /scratch/temp_cherry.pt \
    --model_name_or_path /data/user_data/hdiddee/llama_models/llama_checkpoint/ \
    --max_length 512 \
    --prompt wiz \
    --mod cherry

python ../Cherry_LLM/cherry_seletion/data_by_IFD.py \
    --pt_data_path /scratch/temp_cherry.pt \
    --json_data_path ../sample_data/lima_data.json \
    --json_save_path ../sample_data/cherry_data.json \
    --max_length 512 \
    --sample_number 1000 \
    --prompt wiz


