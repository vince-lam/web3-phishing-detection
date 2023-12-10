from flask import Flask, render_template, request
from utils import get_prediction, load_model_and_tokenizer_for_app

# Initialize Flask app
app = Flask(__name__)


# Define base model, model directory and model filename
BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "models"
model_bin_filename = "pytorch_model.bin"

# Load model and tokenizer for the app
model, tokenizer = load_model_and_tokenizer_for_app(
    model_dir=model_dir, model_filename=model_bin_filename, tokenizer_name=BASE_MODEL
)


# Define the main route for the app
@app.route("/", methods=["GET", "POST"])
def main() -> str:
    try:
        # If the request method is POST, get the text from the form
        if request.method == "POST":
            text = request.form
            messages = text["input"]
            print(messages)

            # Get the prediction for the input text
            full_prediction = get_prediction(model, tokenizer, messages)
            if full_prediction is not None:
                label = full_prediction["LABEL"]
                probability = full_prediction["probability"]
            else:
                label = None
                probability = None

            print(full_prediction)
            # Render the show.html template with the label and probability
            return render_template("show.html", label=label, probability=probability)

        else:
            # If the request method is not POST, render the index.html template
            return render_template("index.html")
    except Exception as e:
        # Render the show.html template with the label and probability

        print(f"An error occurred: {e}")
        return render_template("error.html", error=str(e))


# Run the app
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", debug=True)
    except Exception as e:
        print(f"An error occurred when starting the server: {e}")
