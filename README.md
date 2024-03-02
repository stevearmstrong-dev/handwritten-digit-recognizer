# Creating a Handwritten Digit Recognizer using a Deep Neural Network and measuring its accuracy.

### Introduction
This project focuses on evaluating the accuracy of an image classification model. It involves comparing the model's predictions against the actual labels to determine accuracy.

### Motivation

Image classification is a fundamental task in computer vision, with applications in areas such as object detection, facial recognition, and medical imaging. Developing accurate image classification models is crucial for the reliable performance of these applications. This project aims to contribute to the field of image classification by evaluating the accuracy of a trained model and identifying potential areas for improvement.

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

The dataset used in this project is the MNIST dataset. The MNIST dataset is a standard dataset for handwritten digit recognition. It consists of 70,000 grayscale images of handwritten digits. The images are divided into a training set of 60,000 images and a test set of 10,000 images. The training set is used to train the model, and the test set is used to evaluate the accuracy of the trained model.


### Key Concepts and Techniques:
* DataLoader: Efficiently handles datasets and provides batches of data.
* Transforms: Preprocess and prepare data for training.
* Sequential Model: Simplifies model construction in PyTorch.
* ReLU Activation: Introduces non-linearity, allowing the model to learn complex patterns.
* Cross-Entropy Loss: Measures the performance of classification models whose output is a probability value between 0 and 1.
* SGD Optimizer: Updates model parameters using the gradient of the loss function.
* Backpropagation: Algorithm for training neural networks, involving forward pass, loss computation, backward pass, and weight update.


### Model Architecture

The neural network model defined for this task employs a simple yet effective architecture suitable for learning from the MNIST dataset. The key components of this architecture include:

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

### Summary statistics
print(f"Total images tested: {totalCount}")
print(f"Accurate predictions: {accurateCount}")
print(f"Accuracy percentage: {(accurateCount / totalCount) * 100:.2f}%")

### Results
The script outputs the predicted and actual labels for each image. Upon completion, it displays the total number of images tested, the number of accurate predictions, and the calculated accuracy percentage.

### Future Work
Opportunities for future improvements include refining the model to enhance accuracy further and expanding the test dataset for more comprehensive testing.

## Contact
For any questions or discussions, feel free to contact me at [steve@stevearmstrong.org](mailto:steve@stevearmstrong.org).

### License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/stevearmstrong-dev/handwritten-digit-recognizer/blob/main/LICENSE) file for details.
