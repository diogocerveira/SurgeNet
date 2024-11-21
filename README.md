# SurgeNet

TEMPLATE
## Description
Machine Learning for Surgical Video Phase Analysis
This project aims to develop and evaluate machine learning models, specifically neural networks, to assist in various surgical procedures. Our objective is to enhance precision, improve patient outcomes, and reduce risks through intelligent data analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project environment, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/surgical-nn.git
cd surgical-nn
pip install -r requirements.txt

## Usage
Training the Model
To train the neural network, run the following command:

```bash
python train_model.py --config config.yaml
Running Evaluations
After training, you can evaluate the model performance:

```bash
python evaluate.py --model path/to/model_checkpoint.pth
Data
The dataset consists of annotated surgical videos and images, accessible in the data/processed directory. Ensure that the data is properly preprocessed according to the specifications outlined in data/dataset_preparation.py.

## Model Architecture
This project implements several neural network architectures including:

### Convolutional Neural Networks (CNNs) for image analysis.
### Recurrent Neural Networks (RNNs) for temporal sequence data from surgical videos.
Refer to src/model.py for detailed architecture specifications.

## Training and Evaluation
The training process includes cross-validation to ensure robust model performance. Validation metrics such as accuracy, precision, and recall are logged for analysis. Results can be visualized using TensorBoard.

## Contributing
Contributions are welcome! Please read our Contributing Guidelines for details on the code of conduct, and the process for submitting pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
OpenAI for providing cutting-edge machine learning research and models.
Medical Data Resources for sharing surgical datasets.
Based on:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

https://www.scaler.com/topics/pytorch/how-to-split-a-torch-dataset/

CIFAR10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

Pytorch Forums / Documentation

(hyperparameters default + mostly ChatGPT)
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md
