# Web3 Phishing Detection

An end-to-end pipeline for phishing message detection, encompassing model training and inference, along with a Flask application for user interaction, all containerized using Docker for easy deployment and execution.

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

### Assumptions and Constraints

Assumptions:

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
* Model will output a probability and a threshold will be set to determine the label
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

Due to limited time resource (2.5 days), the following constraints will be applied:

* Will not explore traditional NLP methods and packages, such as nltk and spaCy
* Data validation excluded
* Enforcing typing (e.g. ensure or pydantic) excluded
* Some testing and coverage of methods and functions but not all
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
* Set up DagsHub repo for online experiment and model tracking
* Create an end-to-end training pipeline with a simple transformer (BERT-based) and log metrics with MLflow and DagsHub and
set a baseline. A pipeline consists of:
  * data ingestion
  * data preprocessing
  * model training
  * model evaluation
* Train a few small-sized LLMs from HuggingFace and track experiment metrics due to limited time resources
* Export best performing model as an ONNX or pickle file
* Create Flask app
* Dockerize the container (separate development and production builds)
* Update README

## Set Up

## Model Selection

### Metric results

## Repo Structure

## Tools

## Future Work

With more time, the following features could be explored or implmented:

* the constraints mentioned above
* an ensemble hybrid method using traditional feature engineering and LightGBM
* quantization of the model to reduce model size and improve inference speeds

## License

For this github repository, the License used is MIT License.
