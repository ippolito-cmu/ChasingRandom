python cherry_seletion/data_analysis.py \
    --data_path ../sample_data/lima_data.json \
    --save_path /scratch/temp.pt \
    --model_name_or_path /scratch/<some_base_model> \ 
    --max_length 512 \
    --prompt wiz \
    --mod pre

python cherry_seletion/data_by_cluster.py \
    --pt_data_path /scratch/temp_time.pt \
    --json_data_path lima_full.json \
    --json_save_path lima_full_pre.json \
    --sample_num 10 \
    --kmeans_num_clusters 100 \
    --low_th 25 \
    --up_th 75

python cherry_seletion/data_by_IFD.py \
    --pt_data_path /data/group_data/dei-group/hdiddee/cherry/dolly_cherry_pool_embeddings.pt \
    --model_name_or_path /data/group_data/dei-group/hdiddee/cherry/dolly_pre_experienced_model/ \
    --json_data_path data/flan_data.json \
    --json_save_path time_measure.json \
    --max_length 512 \
    --sample_number 1000 \
    --prompt flan

python cherry_seletion/data_analysis.py \
    --data_path ../sample_data/lima_data.json \
    --save_path /scratch \
    --model_name_or_path /data/group_data/dei-group/hdiddee/cherry/dolly_pre_experienced_model/ \
    --max_length 512 \
    --prompt wiz \
    --mod cherry

