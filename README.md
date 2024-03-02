# Creating a Handwritten Digit Recognizer using a Deep Neural Network and measuring its accuracy with application in Medical Imaging

### Introduction
This project focuses on evaluating the accuracy of an image classification model. It involves comparing the model's predictions against the actual labels to determine accuracy. My focus is on developing a model that reliably classifies handwritten digits, contributing to advancements in computer vision applications like medical imaging.

### Motivation

Image classification serves as a cornerstone in computer vision, underpinning applications from object detection and facial recognition to medical imaging. The goal of this project is to develop a model with high accuracy for these critical applications, thereby supporting the field of image classification.

### Technologies Used
* numpy: To deal with the number lists in the project.
* pyplot from matplotlib: To plot the images from the dataset.
* torch: To compute the tensors and build a deep neural network.
* torchvision: To load datasets.
* transforms from torchvision: To transform a tensor into an array.
* datasets from torchvision: To deal with the dataset.
* optim from torch: To implement the optimization algorithm.
* nn from torch: To take the 3-D transpose of the images.


### Dataset

The dataset used in this project is the MNIST dataset, a standard dataset in handwritten digit recognition comprising 70,000 grayscale images of digits (0-9). It is split into 60,000 training images and 10,000 testing images, enabling us to train and evaluate our model effectively.

![image](https://github.com/stevearmstrong-dev/handwritten-digit-recognizer/assets/113034949/f7ca4e29-b60c-4f24-b505-a00043ffc5c9)


### Key Concepts and Techniques
* DataLoader: Efficiently handles datasets and provides batches of data.
* Transforms: Preprocess and prepare data for training.
* Sequential Model: Simplifies model construction in PyTorch.
* ReLU Activation: Introduces non-linearity, allowing the model to learn complex patterns.
* Cross-Entropy Loss: Measures the performance of classification models whose output is a probability value between 0 and 1.
* SGD Optimizer: Updates model parameters using the gradient of the loss function.
* Backpropagation: Algorithm for training neural networks, involving forward pass, loss computation, backward pass, and weight update.


### Model Architecture

The model features an input layer that flattens 28x28 images into 784-element vectors, two hidden layers with ReLU activation (sizes 64 and 32), and an output layer with 10 units for class probabilities.

* Input Layer: This layer flattens the input images into one-dimensional vectors. Since each MNIST image is 28x28 pixels, the resulting input size for each image is 784 (28*28). This transformation is necessary because the neural network operates on vectors of numbers rather than two-dimensional images.

* Hidden Layers: Two hidden layers are used to process the input data. The first hidden layer transforms the input vector of size 784 to a hidden vector of size 64. The second hidden layer further transforms data from the first hidden layer into another hidden vector of size 32. These layers use the ReLU (Rectified Linear Unit) activation function, which introduces non-linearity to the model, enabling it to learn complex patterns in the data.

* Output Layer: The final layer of the network is the output layer, which consists of 10 neurons corresponding to the 10 possible classes (digits 0-9). This layer transforms the output of the last hidden layer into a probability distribution over the 10 classes, which is used to make predictions.

### Training Process
The training process involves adjusting the model's parameters (weights and biases) to minimize the difference between the predicted and actual labels. This process is iterative and consists of several key steps:

* Forward Propagation: In this step, the model makes predictions based on the current state of its parameters. It involves passing the input data through the model from the input layer to the output layer.

* Loss Calculation: After making predictions, the model calculates the loss using a loss function. In this case, the Cross-Entropy Loss function is used, which is common for classification tasks. This function quantifies how well the model's predictions match the actual labels.

* Backward Propagation: With the loss calculated, the model performs backward propagation (or backpropagation). This step involves calculating the gradient of the loss function concerning each parameter in the model. These gradients indicate how the loss would change with small changes in the parameters and are used to adjust the parameters in the direction that reduces the loss.

* Weight Update: Finally, the model updates its parameters using the gradients calculated during backpropagation. This update is performed using an optimization algorithm, with Stochastic Gradient Descent (SGD) being used in this project. The learning rate, a hyperparameter, controls the size of the updates to prevent the parameters from changing too drastically.

### Epochs and Batches
The training process is conducted over multiple epochs, where one epoch represents a complete pass through the entire training dataset. The dataset is divided into batches (in this case, size 64), and the model's parameters are updated after processing each batch. This approach, known as mini-batch gradient descent, strikes a balance between the computational efficiency of batch gradient descent and the stochastic nature of stochastic gradient descent, leading to more stable and efficient training.

Throughout the training, the model's performance is monitored by calculating the loss and, optionally, other metrics such as accuracy. These metrics provide insight into how well the model is learning and can be used to make adjustments to the training process, such as tuning hyperparameters or modifying the model architecture.

<img width="489" alt="Screenshot 2024-03-02 at 6 36 21 PM" src="https://github.com/stevearmstrong-dev/handwritten-digit-recognizer/assets/113034949/a264c20c-9392-449e-954c-109a121dd500">


### Code Overview

#### Setup and Preprocessing
* Importing Libraries: Essential libraries such as torch, torchvision, numpy, and matplotlib.pyplot are imported to provide the necessary functionalities for data handling, neural network construction, and visualization.
* Data Transformation: The MNIST dataset images are transformed into normalized tensors using transforms.Compose, ensuring they are in the correct format for model input.
#### Data Loading
* Dataset Loading: The MNIST dataset is loaded using datasets.MNIST from torchvision, specifying paths for training and testing datasets.
* DataLoader Use: DataLoader objects for both training and testing sets are created to efficiently manage data batching, shuffling, and iteration during model training and evaluation.
#### Model Definition
* Model Architecture: A Sequential model is defined, incorporating two hidden layers with ReLU activation functions and an output layer designed for classification across ten digit classes.
* Loss Function and Optimizer: Cross-Entropy Loss is chosen for evaluating model predictions, and Stochastic Gradient Descent (SGD) serves as the optimizer, guiding the model's parameter updates.
#### Training Loop
* Batch Processing: For each epoch, the training loop processes batches of images and labels, executing forward propagation, loss calculation, and backward propagation.
* Parameter Updates: Within each iteration, model parameters are updated based on the computed gradients to minimize the loss function.
#### Evaluation and Results
* Testing: After training, the model's performance is assessed on the test dataset, comparing predicted labels against actual labels to calculate accuracy.
* Summary Statistics: Accuracy statistics are printed, providing insight into the model's generalization ability on unseen data.
#### Utilities
* Prediction Function: A utility function, get_predicted_label, is provided to facilitate the prediction of digit classes for individual images, demonstrating the model's practical application.
#### Visualization
* Sample Visualizations: Code snippets for visualizing sample images from the dataset, along with their predicted and actual labels, are included, offering a qualitative assessment of the model's performance.

## Results

<img width="567" alt="Screenshot 2024-03-02 at 6 34 53 PM" src="https://github.com/stevearmstrong-dev/handwritten-digit-recognizer/assets/113034949/4ffeb20d-5fbc-402f-ae99-853a513e075d">

The script outputs the predicted and actual labels for each image. Upon completion, it displays the total number of images tested, the number of accurate predictions, and the calculated accuracy percentage.

## How to Use

This project is designed to be accessible and straightforward to run using Jupyter Notebooks, a popular tool in data science for interactive computing.

### Prerequisites

To run the `handwritten_digit_recognizer.ipynb` notebook, you'll need to have Python installed on your system along with Jupyter Notebook or JupyterLab. It's also recommended to use a virtual environment for Python projects to manage dependencies effectively.

### Installation

1. **Clone the Repository**: Start by cloning this repository to your local machine.
   ```bash
   git clone https://github.com/stevearmstrong-dev/handwritten-digit-recognizer.git
   cd handwritten-digit-recognizer

2. **Create a Virtual Environment (Optional but recommended)**:
   * For **conda** users:
     ``` bash
     conda create --name handwritten-digit-recognizer python=3.8
     conda activate handwritten-digit-recognizer
   * For **venv** users:
     ```bash
     python3 -m venv handwritten-digit-recognizer
     source handwritten-digit-recognizer/bin/activate  # On Windows use `handwritten-digit-recognizer\Scripts\activate`

3. **Install Required Packages**
   ```bash
   pip install numpy matplotlib torch torchvision

4. **Running the Notebook**
   * Navigate to the Notebook Directory: Change directory to the `notebooks` folder.
     ```bash
     cd notebooks
   * Launch Jupyter Notebook
   ```bash
    jupyter notebook
5. **Open `handwritten-digit-recognizer.ipynb` in the Jupyter Notebook interface** and follow the instructions within the notebook to run the analyses.

#### Future Work
Opportunities for future improvements include refining the model to enhance accuracy further and expanding the test dataset for more comprehensive testing.

## Contact
For any questions or discussions, feel free to contact me at [steve@stevearmstrong.org](mailto:steve@stevearmstrong.org).

### License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/stevearmstrong-dev/handwritten-digit-recognizer/blob/main/LICENSE) file for details.
