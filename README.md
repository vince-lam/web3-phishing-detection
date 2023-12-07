# Web3 Phishing Detection

This aim of this project is to build an end-to-end training pipeline to train a phishing message
detection model using a web3 phishing dataset, with a Large Language Model within 3 days.

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

* Positive label is a phishing message
* A false negative (a phishing message not caught) is more detrimental to the business than a false
positive (a genuine message predicted as a phishing attempt)
* In the trade-off between inference speed/latency and accuracy, inference speed is preferred - so
will prefer smaller LLMs
* Model will output a probability and a threshold will be set to determine the label
* Will apply OOP principles but will not fully abstract the codebase
* The LLM is called via an API rather than hosted on-prem
* Assume no fine tuning of LLMs is required
* Due to time constraints, the following features will be excluded:
  * data validation
  * enforcing typing with ensure or pydantic
  * testing and coverage of all methods and functions
  * augmenting the dataset by artificially creating new samples
  * supplementing the dataset with external data

## Roadmap

* Explore dataset and assess data quality
* Define problem statement and evaluation metrics
* Initialise git
* Create repo structure
* Create .gitignore
* Set up pre-commit and github actions CI/CD for linting, sorting, and typing
* Create an end-to-end pipeline with a simple transformer (BERT) and log metrics with MLflow and DagsHub and set a baseline. A pipeline consists of:
  * data ingestion
  * model training
  * model evaluation
* Feature engineer new simple features, such as character count, word count, and suspicious words count, run pipeline and track experiment metrics. Add following step to pipeline:
  * model transformation
* Train multiple small-sized LLMs from HuggingFace and track experiment metrics
* Feature engineer more complex features that require machine learning, such as sentiment analysis, run pipeline and track experiment metrics
* Augment the dataset as it is small for a deep learning problem at ~500 instances and track impact on experiment metrics.
* Create Flask app
* Dockerise the container

## Set Up

## Repo Structure

## Tools

## Future Work

## License

For this github repository, the License used is MIT License.
