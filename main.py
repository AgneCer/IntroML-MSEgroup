import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import yaml
import argparse
import dataset
import model as mdl
from train import train_model
import utils
import os



def main(args):
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["logger"]["wandb"]:
        import wandb
        wandb.login()
        wandb.init(project="UDA", 
                   config=config, 
                   name=f"{args.run_name}")
        
    batch_size_train = config["data"]["batch_size_train"]
    batch_size_test = config["data"]["batch_size_test"]
    output_dim = config["data"]["output_dim"]
    num_workers = config["data"]["num_workers"]
    num_epochs = config["training"]["num_epochs"]
    save_name = config["training"]["save_name"]
    load_model = config["pretrained"]["pre_t"]
    pretrained_model = config["pretrained"]["load"]
    logger = config["logger"]["wandb"]
    current_model = config["model"]
    path_root = config["data"]["path_root"]
    path_root = os.path.normpath(path_root)


    criterion = nn.CrossEntropyLoss()

    # Get model
    if current_model == "CLIP":

        # Install required packages
        utils.install_CLIP('git+https://github.com/openai/CLIP.git')
        
        model, preprocess = mdl.get_CLIP_model(output_dim=output_dim)  # 102 classes in Flowers102 dataset

        if load_model:
            utils.load_model(model, pretrained_model, current_model, output_dim)
            print("Pretrained model loaded!")

        if config["dataset"]=="Flowers":
            train_loader, val_loader = dataset.get_data_flowers(batch_size_train, batch_size_test, num_workers, transform=preprocess)
        
        elif config["dataset"]=="Train_Competition":
            train_loader, val_loader, mean, std  = dataset.create_dataloader(path_root, batch_size_train, img_size=224, val_split=0.2, mode='train', mean=None, std=None, stats_dir='stats', transform=preprocess)
            print("You're training for the competition!")

        else:
            # ADD HERE OTHER DATA LOADERS
            print("Cannot find dataset")

        # Train the model
        # Set up the loss function and optimizer
        optimizer = optim.Adam(model.bottleneck.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, logger, save_name, current_model)

    else:
        if config["dataset"] == "Flowers":
            train_loader, val_loader = dataset.get_data_flowers(batch_size_train, batch_size_test, num_workers)
        
        elif config["dataset"]=="Train_Competition":
            train_loader, val_loader, mean, std  = dataset.create_dataloader(path_root, batch_size_train, img_size=224, val_split=0.2, mode='train', mean=None, std=None, stats_dir='stats', transform=None)
            print("You're training for the competition!")
        else:
            # ADD HERE OTHER DATA LOADERS
            print("Cannot find dataset")

        if current_model == "Vgg19":
            model = mdl.get_Vgg19_model(output_dim=output_dim)

            if load_model:
                utils.load_model(model, pretrained_model, current_model, output_dim)
                print("Pretrained model loaded!")

            optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, logger, save_name, current_model)

        elif current_model == "ResNet50":
            model = mdl.get_ResNet50_model(output_dim=output_dim)

            if load_model:
                utils.load_model(model, pretrained_model, current_model, output_dim)
                print("Pretrained model loaded!")

            optimizer = optim.Adam(model.additional_layers.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, logger, save_name, current_model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--run_name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()
    main(args)