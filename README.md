# Handwritten-Digit-Recognization-using-deep-learning
Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras.


The dataset was constructed from a number of scanned document dataset,
available from the National Institute of Standards and Technology (NIST)

Each image is a 28 by 28 pixel square (784 pixels total).
A standard spit of the dataset is used to evaluate and compare models,where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error,
which is nothing more than the inverted classification accuracy.

NETWORK TOPOLOGY IS SUMMARIZED AS FOLLOWS:
Convolutional layer with 30 feature maps of size 5×5.
Pooling layer taking the max over 2*2 patches.
Convolutional layer with 15 feature maps of size 3×3.
Pooling layer taking the max over 2*2 patches.
Dropout layer with a probability of 20%.
Flatten layer.
Fully connected layer with 128 neurons and rectifier activation.
Fully connected layer with 50 neurons and rectifier activation.
Output layer
