from utilities import get_prediction, load_model_and_tokenizer, read_yaml

config_file_path = "config.yaml"
config = read_yaml(config_file_path)

BASE_MODEL = config["BASE_MODEL"]
output_dir = config["output_dir"]

model, tokenizer = load_model_and_tokenizer(
    model_path=f"./{output_dir}/", tokenizer_name=BASE_MODEL
)

if __name__ == "__main__":
    while True:
        test_input = input(">>> ")
        prediction = get_prediction(model, tokenizer, test_input)
        print(prediction)
