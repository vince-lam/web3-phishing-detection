"""
This script is used for training and evaluating a sequence classification model using the
transformers library. It reads in a configuration file, loads the necessary data and model,
and then trains and evaluates the model.
"""

import os
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utilities import (
    compute_metrics,
    get_prediction,
    load_model_and_tokenizer,
    read_yaml,
    tokenize_data,
)

config_file_path: str = "config.yaml"
config: Dict[str, Any] = read_yaml(config_file_path)

data_file_path: str = config["data_file_path"]
BASE_MODEL: str = config["BASE_MODEL"]
output_dir: str = config["output_dir"]
id2label: Dict[int, str] = config["id2label"]
label2id: Dict[str, int] = config["label2id"]
eval_metric: str = config["eval_metric"]

pytorch_bin_filename: str = config["pytorch_bin_filename"]
model_onnx_filename: str = config["model_onnx_filename"]
experiment_name: str = config["experiment_name"]
full_output_dir: str = f"{output_dir}/{experiment_name}"


def train_and_evaluate(
    model: AutoModelForSequenceClassification,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: AutoTokenizer,
    eval_metric: str,
) -> None:
    """
    Function to train and evaluate a sequence classification model.

    Args:
        model (AutoModelForSequenceClassification): The model to be trained and evaluated.
        train_dataset (Any): The training dataset.
        eval_dataset (Any): The evaluation dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used.
        eval_metric (str): The evaluation metric to be used.

    Returns:
        None
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()


def save_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    path: str = f"./{full_output_dir}/",
) -> None:
    """
    Function to save a sequence classification model and its tokenizer.

    Args:
        model (AutoModelForSequenceClassification): The model to be saved.
        tokenizer (AutoTokenizer): The tokenizer to be saved.
        path (str): The path where the model and tokenizer will be saved.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    # Save the model's state_dict using torch.save
    torch.save(model.state_dict(), f"{path}/{pytorch_bin_filename}")
    # Save the tokenizer using save_pretrained
    tokenizer.save_pretrained(path)


def save_model_as_safetensors(
    model: AutoModelForSequenceClassification, path: str = f"./{full_output_dir}/"
) -> None:
    model.save_pretrained(path)


def convert_to_onnx(
    model: AutoModelForSequenceClassification,
    path: str = f"./{full_output_dir}",
    tokenizer_name: str = BASE_MODEL,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(torch.device("cpu"))

    inputs = tokenizer("Phishing detection", return_tensors="pt")
    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    # Export the model
    with torch.no_grad():
        model.eval()
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            f"{path}/{model_onnx_filename}",
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def main() -> None:
    """
    Main function to train, evaluate, save, and load a sequence classification model. It also
    makes an example prediction.

    The function does the following:
    - Tokenizes the training and evaluation data.
    - Trains and evaluates a sequence classification model.
    - Saves the trained model.
    - Loads the model and tokenizer.
    - Makes an example prediction.

    Returns:
        None
    """
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized_train, tokenized_eval = tokenize_data(data_file_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    train_and_evaluate(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        eval_metric=eval_metric,
    )

    # Save the trained model
    save_model(tokenizer=tokenizer, model=model)

    # Convert to transformers
    # save_model_as_safetensors(model=model)

    # Load the model and tokenizer
    new_model, new_tokenizer = load_model_and_tokenizer()

    # convert_to_onnx(model=model, path=f"./{full_output_dir}", tokenizer_name=BASE_MODEL)

    # Example prediction
    example = """Verify your email address on Binance to unlock additional features and enhance the security of your
    account. Complete the email verification process now and stay in control of your funds and trading activities."""
    print(f"Example message: {example}")
    print("Prediction:")
    print(get_prediction(new_model, new_tokenizer, example))


if __name__ == "__main__":
    main()
