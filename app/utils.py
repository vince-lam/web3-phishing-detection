import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "models"
model_bin_filename = "pytorch_model.bin"


def load_model_and_tokenizer_for_app(
    model_dir=model_dir, model_filename=model_bin_filename, tokenizer_name=BASE_MODEL
):
    # Load the model's state_dict using torch.load
    model_state_dict = torch.load(f"{model_dir}/{model_filename}")
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, state_dict=model_state_dict
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
