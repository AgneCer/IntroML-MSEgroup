
# Image Classification Project

This is the folder for the project of the course "Introduction to Machine Learning" taught by professors Rota, Wang and Liberatori for UniTn, 2024. This model is designed to train and evaluate fine grained image classification models using various datasets. The project includes scripts for data preprocessing, model training, testing, and submission for a final competition.

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

1. **Training the Model:**
   ```bash
   python train.py --config config_flowers102.yaml
   ```

   Replace `config_flowers102.yaml` with the appropriate configuration file for your dataset.

2. **Testing the Model:**
   ```bash
   python test.py --config config_flowers102.yaml
   ```

3. **Generating Submissions:**
   ```bash
   python submit.py --config config_flowers102.yaml
   ```

## Configuration

The configuration files (`config_flowers102.yaml`, `config_Dogs.yaml`, `config_CUB.yaml`, `config_Cars.yaml`) contain parameters for training, testing, and data paths. Customize these files to suit your needs.


## Acknowledgements

This project uses data from the following sources:
- [Flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Stanford Cars Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## Authors

This project was created by the MSE-MagicheSireneEnterprise group, consisting of
 - Agnese Cervino @AgneCer
 - Alessandra Gandini @alegandini
 - Gaudenzia Genoni @Ggenoni
 - Maria Amalia Pelle @pariamelle