from http.client import responses
import requests
import json
import utils
import argparse
import torch
import yaml

def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    model = config["model"]
    filename = None
    current_model = None

    mean = config["param"]["mean"]
    sd = config["param"]["sd"]
    #test_loader = (PASS mean and sd)

    best_model = utils.load_model(model, filename, current_model)
    
    # `keep_full_label=True` for complete lable (id_name)
    # `keep_full_label=False` to get just the id
    preds = utils.return_predictions_dict(best_model, test_loader, device, keep_full_label=False)

    res = {
        "images": preds,
        "groupname": "MSE-MagnificheSireneEnterprise"
    }

    submit(res)