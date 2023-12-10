import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
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

config_file_path = "config.yaml"
config = read_yaml(config_file_path)

data_file_path = config["data_file_path"]
BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]
id2label = config["id2label"]
label2id = config["label2id"]
pytorch_bin_filename = config["pytorch_bin_filename"]
model_onnx_filename = config["model_onnx_filename"]


def train_and_evaluate(model, train_dataset, eval_dataset, tokenizer):
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
        metric_for_best_model="f1",
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


def save_model(model, tokenizer, path=f"./{output_dir}/"):
    # Save the model's state_dict using torch.save
    torch.save(model.state_dict(), f"{path}/{pytorch_bin_filename}")
    # Save the tokenizer using save_pretrained
    tokenizer.save_pretrained(path)


def save_model_as_safetensors(model, path=f"./{output_dir}/"):
    model.save_pretrained(path)


def save_model_as_pytorch(model, tokenizer, path=f"./{output_dir}/"):
    # Save the model's state_dict using torch.save
    torch.save(model.state_dict(), f"{path}/{pytorch_bin_filename}")
    # Save the tokenizer using save_pretrained
    tokenizer.save_pretrained(path)


def convert_to_onnx(model, path=f"./{output_dir}", tokenizer_name=BASE_MODEL):
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


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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
    )

    # Save the trained model
    save_model(tokenizer=tokenizer, model=model)

    # Convert to transformers
    save_model_as_safetensors(model=model)

    # Load the model and tokenizer
    new_model, new_tokenizer = load_model_and_tokenizer()

    convert_to_onnx(model=model, path=f"./{output_dir}", tokenizer_name=BASE_MODEL)

    # Example prediction
    example = """Verify your email address on Binance to unlock additional features and enhance the security of your
    account. Complete the email verification process now and stay in control of your funds and trading activities."""
    print(get_prediction(new_model, new_tokenizer, example))


if __name__ == "__main__":
    main()
