from http.client import responses
import requests
import json
import utils
import argparse
import torch
import yaml
import dataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import model as mdl
from train import train_model

def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


parser = argparse.ArgumentParser("")
parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
args = parser.parse_args()

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
picked_model = config["model"]
output_dim= config["data"]["output_dim"]
filename = config["pretrained"]["load"]



# Pick model
if  picked_model == "CLIP":
    utils.install_CLIP('git+https://github.com/openai/CLIP.git')
    best_model, preprocess = mdl.get_CLIP_model(output_dim=output_dim)  
    test_loader = None


else:
    test_loader = None
    if picked_model == "Vgg19":
        best_model = mdl.get_Vgg19_model(output_dim=output_dim)
    elif picked_model == "ResNet50":
        best_model = mdl.get_ResNet50_model(output_dim=output_dim)
    else: 
        print("something went wrong in uploading ResNet or Vgg")

model_with_weight = utils.load_model(best_model, filename, picked_model)

# `keep_full_label=True` for complete lable (id_name)
# `keep_full_label=False` to get just the id
preds = utils.return_predictions_dict(model_with_weight, test_loader, device, keep_full_label=False)

res = {
    "images": preds,
    "groupname": "MSE-MagnificheSireneEnterprise"
}

submit(res)

