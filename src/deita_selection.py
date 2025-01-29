# Please refer to https://github.com/hkust-nlp/deita to see usage instructions, argument descriptions and to see options for pre-trained scorers.
from deita.pipeline import Pipeline
import os
import argparse
import time
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default=0.9, type=float)
    parser.add_argument(
        "--full_dataset_file", default="./sample_data/deita_alpaca.jsonl", type=str
    )
    parser.add_argument("--deita_data", default="./deita_alpaca.jsonl", type=str)
    parser.add_argument("--deita_embedding_path", default="/scratch", type=str)
    args = parser.parse_args()

    print(f"{args.full_dataset_file} is being processed ...")

    dataset = args.full_dataset_file.split("_")[0].replace(".jsonl", "")
    print(f"Processing {dataset} ... Starting timer ...")
    start_time = time.time()

    embedding_model_path = os.path.join(
        args.deita_embedding_path, "embeddings", f"{dataset}.pkl"
    )

    embed_pipeline = Pipeline(
        "embed_pipeline",
        data_path=os.path.join(
            args.full_dataset_file
        ),  # json file with sharegpt format
        output_path=embedding_model_path,  # output path (pickle format)
        model_name_or_path="mistralai/Mistral-7B-v0.1",  # model name or path e.g. mistralai/Mistral-7B-v0.1
        max_length=512,
        use_flash_attention=True,
        conv_template="vicuna_v1.1",
        batch_size_per_device=2,
        bfloat16=True,
    )

    embed_pipeline.run()
    print(
        f"Embedding Construction Successful ... Time taken for embedding construction: {(time.time() - start_time) / 60} minutes."
    )
    del embed_pipeline
    gc.collect()
    start_time = time.time()

    pipeline = Pipeline(
        "score_pipeline",
        data_path=os.path.join(args.full_dataset_file),
        scorer="llama",  # Can utilize the trained scorers from deita's official repository as well. Find all options at https://github.com/hkust-nlp/deita
        scorer_name_or_path="hkust-nlp/deita-quality-scorer",
        is_vllm=False,
        score_type="quality",
        output_path=os.path.join(args.deita_data, "scores", f"{dataset}.json"),
    )
    pipeline.run()
    print(
        f"Scoring Successful for {dataset} ... Time taken for scoring: {(time.time() - start_time) / 60} minutes. Moving on to filtering ..."
    )
    del pipeline
    gc.collect()

    start_time = time.time()

    filter_pipeline = Pipeline(
        "filter_pipeline",
        data_path=os.path.join(
            args.deita_data, "scores", f"{dataset}.json"
        ),  # json file with sharegpt format
        other_data_path=embedding_model_path,  # embedding file path (pickle format)  # filter threshold default: 0.9
        data_size=10000,
        sort_key="quality_scores",  # default: "complexity_scores,quality_scores"
        chunk_size=100000,
        threshold=args.threshold,
        embedding_field="embedding",
        distance_metric="cosine",
        output_path=os.path.join(
            args.deita_data, f"{dataset}_filtered_data.json"
        ),  # json format output path
        device=0,
    )
    filter_pipeline.run()
    print(
        f"Filtering Successful for {dataset} ... Time taken for filtering: {(time.time() - start_time) / 60} minutes."
    )
    del filter_pipeline
    gc.collect()
