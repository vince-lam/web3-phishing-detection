import time

import evaluate
import numpy as np
from evaluate import evaluator
from transformers import pipeline
from utilities import load_model_and_tokenizer, prepare_data, read_yaml, tokenize_data

config_file_path = "config.yaml"
config = read_yaml(config_file_path)

data_file_path = config["data_file_path"]
BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]
model_metrics_filename = config["model_metrics_filename"]
eval_metric = config["eval_metric"]
experiment_name = config["experiment_name"]
full_output_dir: str = f"{output_dir}/{experiment_name}"


def get_metrics():
    _, tokenized_eval = tokenize_data(data_file_path)

    new_model, new_tokenizer = load_model_and_tokenizer(
        model_path=f"./{full_output_dir}/", tokenizer_name=BASE_MODEL
    )

    clf_metrics = evaluate.combine(
        [
            "f1",
            "recall",
            "precision",
            "accuracy",
        ]
    )

    pipe = pipeline("text-classification", model=new_model, tokenizer=new_tokenizer, device=-1)
    task_evaluator = evaluator("text-classification")

    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=tokenized_eval,
        metric=clf_metrics,
        label_mapping={"GENUINE": 1, "PHISHING": 0, "NEGATIVE": 0, "POSITIVE": 1},
    )

    return results


def save_results(file_name=f"{full_output_dir}/{model_metrics_filename}"):
    with open(file_name, "a") as f:
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write(f"Fine tuned base model: {BASE_MODEL}\n")
        f.write(f"Evaluated on: {eval_metric}\n")
        f.write(f"Results: {str(get_metrics())}\n\n")
        f.close()


def main():
    save_results()
    print("Finished evaluating model.")


if __name__ == "__main__":
    main()
