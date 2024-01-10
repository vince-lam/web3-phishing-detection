import os

import evaluate
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_config():
    config_file_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    return read_yaml(config_file_path)


config = get_config()

BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]
pytorch_bin_filename = config["pytorch_bin_filename"]
eval_metric = config["eval_metric"]
experiment_name = config["experiment_name"]
full_output_dir = f"{output_dir}/{experiment_name}"


def load_data(data_file_path):
    data = pd.read_csv(data_file_path)
    data = data.rename(columns={"Messages": "text", "gen_label": "label"})

    return data


def prepare_data(data_file_path, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        load_data(data_file_path), test_size=test_size, random_state=random_state
    )
    # Convert the Pandas DataFrames into Hugging Face Datasets
    train = Dataset.from_pandas(train_df)
    test = Dataset.from_pandas(test_df)
    train = train.remove_columns("__index_level_0__")
    test = test.remove_columns("__index_level_0__")

    return train, test


def combine_datasets(train, test):
    dataset_dict = DatasetDict(
        {
            "train": train,
            "test": test,
        }
    )

    return dataset_dict


def tokenize_data(data_file_path, model=BASE_MODEL, test_size=0.2, random_state=42):
    train, test = prepare_data(data_file_path, test_size=test_size, random_state=random_state)

    tokenizer = AutoTokenizer.from_pretrained(model)

    def preprocess_function(examples):
        # Create a preprocessing function to tokenize text and truncate sequences to be no longer than maximum input length
        return tokenizer(examples["text"], truncation=True)

    # Apply the preprocessing function over the entire dataset
    tokenized_train = train.map(preprocess_function, batched=True)
    tokenized_eval = test.map(preprocess_function, batched=True)

    return tokenized_train, tokenized_eval


def compute_metrics(eval_pred, metric_choice=eval_metric):
    metric = evaluate.load(metric_choice)

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def load_model_and_tokenizer(model_path=f"./{full_output_dir}/", tokenizer_name=BASE_MODEL):
    # Load the model's state_dict using torch.load
    model_state_dict = torch.load(f"{model_path}/{pytorch_bin_filename}")
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, state_dict=model_state_dict
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_prediction(model, tokenizer, text):
    encoding = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    )
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    outputs = model(**encoding)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
    label = np.argmax(probs, axis=-1)

    return {
        "LABEL": "GENUINE" if label == 1 else "PHISHING",
        "probability": probs[1] if label == 1 else probs[0],
    }
