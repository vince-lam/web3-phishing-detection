# Web3 Phishing Detection

An end-to-end pipeline for phishing message detection, encompassing model training and inference, along with a Flask application for user interaction, all containerized using Docker for easy deployment and execution. Built in 2 days.

## Background

The rapid evolution of blockchain technology and the rise of decentralized applications
(dApps) have given birth to the Web3 ecosystem. While this technology promises greater
control and ownership over digital assets, it has also introduced new security challenges.
One of the most significant concerns is the prevalence of phishing attacks targeting Web3
users with the intent of obtaining their seed phrases and subsequently draining their wallets
for fraudulent activities

Web3 phishing attacks are a critical issue in the modern blockchain landscape. These
attacks involve the use of deceptive and fraudulent tactics to manipulate users into revealing
their sensitive information, primarily their seed phrases. Seed phrases, also known as
mnemonic phrases or recovery phrases, are sets of words that act as a cryptographic key to
access and control users' wallets and assets within the Web3 ecosystem.

The core problem lies in the fact that users often lack awareness and knowledge about the
security practices required to protect their seed phrases. Phishing attackers exploit this
vulnerability by deploying convincing phishing messages through various communication
channels, including email, social media, messaging apps, and even fake dApps. These
messages typically prompt users to click on malicious links, enter their seed phrases on
counterfeit websites, or share their confidential information.

## Set Up

To run the Flask app:

1. Install docker
2. In the terminal run: `docker run -p8888:8888 vincenthml/ml-app`
3. Go to `http://127.0.0.1:8888` or `http://172.17.0.2:8888` in a web browser

To run the code in this repo:

1. Clone this repo and cd to the root directory
2. Create a new virtual environment, I recommend using venv by running:

* `python -m venv .venv`
* `source .venv/bin/activate`
* `python -m pip install --upgrade pip`

3. Install poetry by running: `pip install poetry`
4. Install python packages with poetry by running: `poetry install`

Now you can do the following:

1. Download models from HuggingFace, train them on phishing dataset, save the new model and tokenizers

* First, copy the phishing dataset to the `ml_dev/data` directory
* Open `ml_dev/config.yaml` and update `data_file_path`, `BASE_MODEL`, and other parameters desired
* Run `cd ml_dev`
* Run `python train_and_save_model.py`

2. Evaluate the newly trained model on f1, recall, precision, and accuracy

* In `ml_dev` directory, run `python generate_model_metrics.py`
* This will create a txt file specified in the `config.yaml` at the path of `{output_dir}/{experiment_name}`

3. Test the model predictions in the CLI

* In `ml_dev` directory, run `python test_predictions.py`
* Enter messages in terminal to run predictions

To create a new docker image so the Flask app can run an improved model:

1. Make sure the best performing model and associated tokenizer files are saved in `app/models` and remove all other models
2. `cd app`
3. Update `image` parameter in `docker-compose.yaml` to correct username and project name
4. Build the docker image: `docker compose build up --build`
5. Push image to Docker Hub for reproducibility: `docker compose push`

### Deliverables

* Working code for the end-to-end training pipeline using Python scripts.
* Working code for loading the trained model and performing inference using Python scripts.
* The end-to-end training pipeline must be implemented with Python scripts and not in a Jupyter notebook.
* Evaluate your trained model and document the metric results in a README.
* Working code for the Flask application.
* README with instructions on how to execute the training pipeline.
* README on how to run the packaged Flask application. (NOTE: The Flask application must be runnable
after packaging, and it should have an endpoint that can be queried using a POST method.)
* A requirements.txt file for all dependencies.
* Dockerfile(s) or docker-compose.yaml file to start the frontend and backend application.
* Lastly, package the model as a Flask application and provide a simple UI that allows users
to interact with it. The output should be either binary or the probability score of the message
being a phishing message.

### Assumptions

* Metrics
  * An instance with `gen_label` of 1 means that it is a genuine message
  * However, assuming convention where a phishing attempt is labelled as `y=1` and a geuine message is labelled as `y=0`, then a false negative (a phishing message not caught) is more detrimental to the business than a false
positive (a genuine message predicted as a phishing attempt)
  * Success metric is F1-score because both false negatives (real phishing attempts) and false positives (genuine
  messages labelled as phishing) are detrimental to the customer, although false negatives are more detrimental.
  * Assume that this model is deployed across all communication channels such as email social media, and messaging apps.
  In reality, you would likely have a different model for each communication channel for different metrics, e.g. recall
  for social media.
* In the trade-off between inference speed/latency and accuracy, inference speed is preferred - so
will prefer smaller LLMs. Added benefit of greater explainability and lower complexity
* Model will output a probability and a threshold will be set to determine the label rather than a binary 1/0. Using a probability score has the added benefit of setting a threshold to catch messages with low certainty, so human labelling can be used to improve dataset quality.
* The model is downloaded and stored on from Docker image rather than hosted on HuggingFace and called via API. This
means the model can be accessed offline, have faster inference speeds, and not rely on internet connection.
* Flask app is deployed locally and not hosted on cloud services such as Heroku or AWS
* Feature engineering, such as character count, word count, suspicious words count, sentiment analysis, is not required
for large language models, as they have been trained on sufficient data to understand context
* Will explore existing pre-trained models which were trained for phishing/spam classification tasks and experiment if
this transfer learning provides greater performance
* Assume that no money is allowed to be spent on this project, so no calling of OpenAI's API and fine tuning of LLMs
using cloud GPU services, such as RunPod
* Assume no class rebalancing is required as the labels are relatively balanced (292:212).
* Assume k-folds cross-validation is not required due to effectiveness of transfer learning for deep learning models

### Constraints

Due to limited time resource (1.5 days), the following constraints will be applied:

* Will not explore traditional NLP methods and packages, such as nltk and spaCy
* Data validation excluded
* Enforcing typing (e.g. ensure or pydantic) excluded
* Some unit testing and coverage of methods and functions but not all
* Limited documentation to classes and functions
* Some logging but throughout the codebase
* Some object-orientated programming principles will be applied but not to full codebase
* Simple Flask app UI
* Explainability of model not explored
* No extensive hyperparameter tuning with optuna

The following MLOps best practices will not be applied:

* Data version control as the dataset is static
* Continuous model deployment CI/CD with GitHub actions or Kubeflow
* Monitoring of model in production is not required, i.e. data drift

## Roadmap

* Explore dataset and assess data quality, e.g. duplicates
* Define problem statement and evaluation metrics
* Initialise git, create repo structure, create .gitignore
* Set up pre-commit and github actions CI/CD for linting (black), sorting (isort), and typing - for standardised styling
* Create an end-to-end training pipeline with a simple transformer (BERT-based) and log metrics to set a baseline. A pipeline consists of:
  * data ingestion
  * data preprocessing
  * model training
  * model evaluation
* Train a few small-sized LLMs from HuggingFace and track experiment metrics due to limited time resources
* Export best performing model as a bin file
* Create Flask app
* Dockerize the container (separate development and production builds)
* Update README

## Model Selection

[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) was chosen as a base model to be further fine tuned using custom web3 phishing dataset using a 80/20 train/test split.

The original dataset contained 30 duplicate pairs which were removed before training.

### Metric results

The model trained using the deduplicated phishing data achieved the following metrics (the metrics of the model trained on the datasets with duplicates is shown within the parenthesis):

* F1 score: 0.906 (0.884)
* Recall: 0.930 (0.897)
* Precision: 0.883 (0.871)
* Accuracy: 0.884 (0.842)
* Latency (seconds): 0.0190 (0.0199)
* Throughput (samples per second): 52.5 (50.3)

## Repo Structure

```markdown
.
├── LICENSE
├── README.md
├── app                                       # Directory for Flask app deployment
│   ├── Dockerfile
│   ├── app.py
│   ├── docker-compose.yaml
│   ├── models
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   ├── requirements.txt
│   ├── templates
│   │   ├── index.html
│   │   └── show.html
│   └── utils.py
├── ml_dev                                    # Directory for model development
│   ├── config.yaml
│   ├── data
│   │   ├── DS test_data.csv
│   │   └── DS test_data_deduped.csv
│   ├── generate_model_metrics.py
│   ├── logs
│   ├── model_outputs
│   ├── notebooks
│   │   └── exploration.ipynb
│   ├── requirements_poetry.txt
│   ├── test_predictions.py
│   ├── train_and_save_model.py
│   └── utilities.py
├── poetry.lock
├── pyproject.toml
└── tests
    └── ml_dev
        └── test_generate_model_metrics.py
```

## Future Work

With more time, the following features could be explored or implemented:

* the constraints mentioned above
* test the following models fine tuned for spam classification and compare results to trained model:
  * <https://huggingface.co/mshenoda/roberta-spam>
  * <https://huggingface.co/Huaibo/phishing_bert>
  * <https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection>
  * <https://huggingface.co/tony4194/distilbert-spamEmail>
* an ensemble hybrid method using traditional feature engineering and LightGBM
* quantization of the model to reduce model size and improve inference speeds

## License

For this github repository, the License used is MIT License.
