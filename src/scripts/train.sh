BUDGET_ROOT="/data/user_data/hdiddee/llama_data/budgets/" 
ROOT="/scratch/"
STRATEGIES=("deita"
            "cherry"
            "longest"
            "alpacasus"
            "random"
            "strictrandom")

BUDGET=(
    "100"
    "1000"
    "10000"
)

for strategy in "${STRATEGIES[@]}"; do
    echo "Searching for files with strategy: $strategy"
    TRAIN_FILE_ARRAY=($(find $BUDGET_ROOT -name "*${strategy}*.jsonl"))

    if [ ${#TRAIN_FILE_ARRAY[@]} -eq 0 ]; then
        echo "No files found for strategy $strategy."
    else
        echo "Files found: ${TRAIN_FILE_ARRAY[@]}"
    fi

    for i in "${!TRAIN_FILE_ARRAY[@]}"; do
        for j in "${!BUDGET[@]}"; do

            train_file="${TRAIN_FILE_ARRAY[$i]}"
            filename=$(basename "${TRAIN_FILE_ARRAY[$i]}")

            base_name=$(basename "$filename" .jsonl)
            number="${filename%.*}" 
            
            IFS='_' read -ra parts <<< "$number"
            num_parts=${#parts[@]}

            # Check if the trial number exists
            if [ "$num_parts" -ge 4 ]; then
                trial="${parts[3]}"
            else
                trial=0
            fi


            # Extract parts of the filename
            dataset=$(echo "$base_name" | cut -d'_' -f1)
            budget=${BUDGET[$j]}

            if [[ "$strategy" == "random" || "$strategy" == "strictrandom" ]]; then
                output_dir="${ROOT}${dataset}/${strategy}/${budget}_${trial}"
                wandb_run_name="${dataset}_${strategy}_${budget}_${trial}"
            else
                output_dir="${ROOT}${dataset}/${strategy}/${budget}"
                wandb_run_name="${dataset}_${strategy}_${budget}"
            fi
            
            echo "File: $train_file"
            echo "Dataset: $dataset"
            echo "Strategy: $strategy"
            echo "Output Dir: $output_dir"
            echo "Wandb Run Name: $wandb_run_name"
            echo "Budget: $budget"
            echo "" 

            accelerate launch --num_processes 8 ../train.py \
                --model_type llama \
                --model_name /data/user_data/hdiddee/llama_models/llama_checkpoint/ \
                --wandb_run_name $wandb_run_name \
                --wandb_project_name "test-run" \
                --do_train \
                --num_train_epochs 3 \
                --budget $budget \
                --train_file $train_file \
                --save_strategy no \
                --learning_rate 2e-5 \
                --per_device_train_batch_size 2 \
                --gradient_accumulation_steps 8 \
                --optim paged_adamw_32bit \
                --weight_decay 0.0 \
                --fp16 False \
                --bf16 True \
                --save_total_limit 1 \
                --output_dir $output_dir \
                --warmup_ratio 0.03 \
                --lr_scheduler_type linear
        done
    done
done
