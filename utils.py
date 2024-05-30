import os
import torch
import subprocess
import sys
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def install_CLIP(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("CLIP installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing CLIP: {e}")
  

def save_model(model, filename, current_model):
    print("Saving models ...")
    save_folder = "trained_model"
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_folder_model = os.path.join(save_folder, current_model)
    if not os.path.exists(save_folder_model):
        os.makedirs(save_folder_model)

    save_path = os.path.join(save_folder_model, filename)
    torch.save(model.state_dict(), save_path)
    print("The model has been successfully saved!")



def load_model(model_class, filename, current_model, output_dim, device=device):
    print("Loading model ...")

    save_folder = "trained_model"
    
    save_folder_model = os.path.join(save_folder, current_model)
    save_path = os.path.join(save_folder_model, filename)
    print(f"Saving path to download model: {save_path}")

    if os.path.exists(save_path):
        model = model_class(output_dim=output_dim).to(device)  # Initialize the model before loading the state
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()  # Put the model in evaluation mode
        print("The model has been successfully loaded!")
        return model
    else:
        print("Model file not found!")
        return None



# predictions on the TEST set
def return_predictions_dict(model, test_loader, device, class_mapping, keep_full_label=True):
    model.eval()
    preds_dict = {}
    
    with torch.no_grad():
        for inputs, image_ids in tqdm(test_loader, total=len(test_loader), desc="Making Predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Iterate through the batch to associate each image_id with the corresponding prediction
            for image_id, label in zip(image_ids, predicted):
                label = label.item()
                class_name = class_mapping[str(label)]  # Get class name from the label
                if not keep_full_label:
                    class_name = class_name.split('_')[0]
                
                preds_dict[image_id] = class_name

    return preds_dict


#use azure machine:
#type into powershell terminal:
#scp -r -P 5012 "D:/DATA SCIENCE/MACHINE LEARNING/CLIP_project" disi@lab-b19fb86e-17c2-41af-aa77-c4a6adf27da4.westeurope.cloudapp.azure.com:/home/disi/
#then copy azure code into ssh terminal
#cd to project folder
#run: python main-py --config config_flowers102.yaml --run_name my_run


#pushing commits to github (terms in uppercase must be changed):
# cd to project folder
# git add . (add files in the staging area)
# git commit -m "message" (commit changes)
# if needed: git remote add NAME_REMOTE https://github.com/USER_NAME/REPO_NAME.git
# git push NAME_REMOTE BRANCH_NAME (or push -f to force changes)