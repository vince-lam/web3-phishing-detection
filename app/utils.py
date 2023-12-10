import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define base model, model directory and model filename
BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "models"
model_bin_filename = "pytorch_model.bin"


def load_model_and_tokenizer_for_app(
    model_dir=model_dir, model_filename=model_bin_filename, tokenizer_name=BASE_MODEL
):
    """
    Load model and tokenizer for the app.

    Parameters:
    model_dir (str): Directory where the model is stored.
    model_filename (str): Filename of the model.
    tokenizer_name (str): Name of the tokenizer.

    Returns:
    model: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    # Load the model's state_dict using torch.load
    model_state_dict = torch.load(f"{model_dir}/{model_filename}", map_location="cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_name, state_dict=model_state_dict
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def get_prediction(model, tokenizer, text):
    """
    Get prediction for the given text.

    Parameters:
    model: The model to use for prediction.
    tokenizer: The tokenizer to use for prediction.
    text (str): The text to predict.

    Returns:
    dict: A dictionary with the label and probability of the prediction.
    """
    # Tokenize the text
    encoding = tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    )
    encoding = {k: v.to(model.device) for k, v in encoding.items()}
    # Get the outputs from the model
    outputs = model(**encoding)
    sigmoid = torch.nn.Sigmoid()
    # Get the probabilities from the outputs
    probs = sigmoid(outputs.logits.squeeze().cpu()).detach().numpy()
    label = np.argmax(probs, axis=-1)

    return {
        "LABEL": "GENUINE" if label == 1 else "PHISHING",
        "probability": probs[1] if label == 1 else probs[0],
    }
