from flask import Flask, render_template, request

from app.utils import get_prediction, load_model_and_tokenizer_for_app

app = Flask(__name__)


BASE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
model_dir = "models"
model_bin_filename = "pytorch_model.bin"

model, tokenizer = load_model_and_tokenizer_for_app(
    model_dir=model_dir, model_filename=model_bin_filename, tokenizer_name=BASE_MODEL
)


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        text = request.form
        # print(text)
        messages = text["input"]
        print(messages)

        full_prediction = get_prediction(model, tokenizer, messages)
        label = full_prediction["LABEL"]
        probability = full_prediction["probability"]

        print(full_prediction)
        return render_template("show.html", label=label, probability=probability)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
