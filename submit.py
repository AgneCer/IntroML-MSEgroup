# Ensure the necessary imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import requests
import argparse
from torchvision import models
import utils
import dataset
import model as mdl

def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# Function to load the class mapping from a JSON file
def load_class_mapping(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

parser = argparse.ArgumentParser("")
parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
args = parser.parse_args()

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

picked_model = config["model"]
output_dim = config["data"]["output_dim"]
filename = config["pretrained"]["load"]
path_root = config["data"]["path_root"]
path_root = os.path.normpath(path_root)
batch_size_test = config["data"]["batch_size_test"]
output_dim = config["data"]["output_dim"]

# Define a function to get the model class
def get_model_class(model_name):
    if model_name == "ResNet50":
        return mdl.get_ResNet50_model
    elif model_name == "Vgg19":
        return mdl.get_Vgg19_model
    elif model_name == "CLIP":
        utils.install_CLIP('git+https://github.com/openai/CLIP.git')
        return mdl.get_CLIP_model
    else:
        raise ValueError("Unsupported model")

# Load the model
model_class = get_model_class(picked_model)
best_model = model_class(output_dim)  # Initialize the model

# Load state dictionary into the initialized model
best_model = utils.load_model(model_class, filename, picked_model, output_dim, device)

if best_model is None:
    print("Error: the model was not loaded correctly.")
else:
    print("The model was loaded correctly.")

# Load the class mapping
class_mapping = load_class_mapping('class_mapping.json')

# Prepare the dataloader
if picked_model == "CLIP":
    _, preprocess = model_class(output_dim=output_dim)
    test_loader = dataset.create_dataloader(path_root, batch_size_test, img_size=224, val_split=0.2, mode='test', mean=None, std=None, stats_dir='stats', transform=preprocess)
else:
    test_loader = dataset.create_dataloader(path_root, batch_size_test, img_size=224, val_split=0.2, mode='test', mean=None, std=None, stats_dir='stats', transform=None)

# Keep full label=True for complete label (id_name)
# Keep full label=False to get just the id
preds = utils.return_predictions_dict(best_model, test_loader, device, class_mapping)

res = {
    "images": preds,
    "groupname": "MSE-MagnificheSireneEnterprise"
}

print(res)
#submit(res)