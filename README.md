
# Fine-Grained Image Classification Project

This is the repository for the project of the course "Introduction to Machine Learning" taught by Professors Rota, Wang and Liberatori for UniTn, 2024. This model is designed to train and evaluate fine-grained image classification models using various datasets. The project includes scripts for data preprocessing, model training, testing, and submission for a final competition.

## Project Structure

- `main.py`: The main entry point for the project. It handles the overall workflow.
- `model.py`: Contains the model architecture and related functions.
- `dataset.py`: Handles data loading and preprocessing.
- `utils.py`: Utility functions used throughout the project.
- `train.py`: Script for training the model.
- `test.py`: Script for evaluating the model on the test dataset.
- `submit.py`: Generates submission files for competition.
- `config_flowers102.yaml`: Configuration file for the Flowers 102 dataset.
- `config_Dogs.yaml`: Configuration file for the Dogs dataset.
- `config_CUB.yaml`: Configuration file for the CUB-200-2011 dataset.
- `config_Cars.yaml`: Configuration file for the Stanford Cars dataset.
- `config_compTrain.yaml`: Configuration file for training the model for the competition.
- `config_compTest.yaml`: Configuration file for testing the model for the competition..

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.1
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AgneCer/IntroML-MSEgroup.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Datasets

Ensure you have the datasets downloaded and available in the appropriate directories. Modify the configuration files if needed to point to the correct paths.

### Running the Project

This code was used for two purposes: to evaluate how three different models (Vgg19, ResNet50 and CLIP) perform in the fine-grained image classification task, and to participate in an internal course competition. Belong is an explanation of how to use the code. 

1. **Training and Testing the models on datasets:**
   To compute train and test datasets simply run the `main.py` file. This, after dividing the dataset into train and validation, will train the model selected in the configuration file for as many epochs as specified, then return the accuracy of both training and validation.

   ```bash
   python main.py --config [choosen_config].yaml --run_name [RunName]
   ```

   Replace `[choosen_config]` with the appropriate configuration file for your dataset and `[RunName]` with the choosen run name. 

2. **Training and Testing the models for the competition:**
    For the course competition, a few changes need to be made to the code run. The initial part remains unchanged: by running `main.py`, and properly configuring the configuration file, the chosen model will examine the dataset, dividing it into train set and validation set, and train it on the available data. 

   ```bash
   python main.py --config config_compTrain.yaml
   ```
    To test the model, and consequently submit the results, yit is necessary to run the `submit.py` file. This loads the previously trained model, and classifies the images in the test set, producing a dictionary where the keys are the IDs of the images and the values are the assigned class.

   ```bash
   python submit.py --config compTest.yaml
   ```

## Configuration
1. **Training and Testing the models on datasets:**
    The configuration files (`config_flowers102.yaml`, `config_Dogs.yaml`, `config_CUB.yaml`, `config_Cars.yaml`) contain parameters for training, testing, and data paths. In particular, they contains:
    - `num_epochs`: number of epochs for training.
    - `save_name`: model save name.
    - `dataset`: name of the dataset.
    - `model`: model to use (ResNet50, Vgg19, or CLIP).
    - `pre_t`: True or False, depending on whether to use a previously trained model.
    - `load`: if the previous is True, it should contain the name of the pre-trained model.
    - `batch_size_train`: batch size for training.
    - `batch_size_test`: batch size for testing.
    - `num_workers`: number of workers used.
    - `output_dim`: dimension of the output, i.e., the number of classes.
    - `path_root`: path to the dataset.
    - `wandb`: True or False, depending on whether the user wants to visualize graphs on WandB.
    
2. **Training and Testing the models for the competition:**
        The configuration files (`config_compTrain.yaml`, `compTest.yaml`) contain parameters for training, testing, and data paths. In particular, they contains:
    - `num_epochs`: number of epochs for training.
    - `save_name`: model save name.
    - `dataset`: name of the dataset.
    - `model`: model to use (ResNet50, Vgg19, or CLIP).
    - `pre_t`: True for the `compTest.yaml`, False for the `config_compTrain.yaml`.
    - `load`: for the `compTest.yaml`, it contains the name of the pretrained model. 
    - `batch_size_train`: batch size for training.
    - `batch_size_test`: batch size for testing.
    - `num_workers`: number of workers used.
    - `output_dim`: dimension of the output, i.e., the number of classes.
    - `path_root`: path to the dataset.
    - `wandb`: True or False, depending on whether the user wants to visualize graphs on WandB.

## Acknowledgements

This project uses data from the following sources:
- [Flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Stanford Cars Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Authors

This project was created by the MSE-MagicheSireneEnterprise group, consisting of:
 - Agnese Cervino - [@AgneCer](https://github.com/AgneCer)
 - Alessandra Gandini - [@alegandini](https://github.com/alegandini)
 - Gaudenzia Genoni - [@Ggenoni](https://github.com/Ggenoni)
 - Maria Amalia Pelle - [@pariamelle](https://github.com/pariamelle)

 