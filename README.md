# DeepLearningA1Final
FeedForward NN and BackPropogation
Fashion MNSIT Dataset:

1. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples.
2. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
3. 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
4. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
5. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing.
6. Each training and test example is assigned to one of the following labels:

      0 T-shirt/top
      1 Trouser
      2 Pullover
      3 Dress
      4 Coat
      5 Sandal
      6 Shirt
      7 Sneaker
      8 Bag
      9 Ankle boot
      
FeedForward NN:
In the fashion_mnist train dataset n=784, m=60000. Where n is the number of pixels in one image and 60000 is the number of example images available for training. Of this we keep 10% of the examples for validating if our model had trained properly. Therefore there are 54000 training data and 6000 validation data examples. 

In getData function of the code we initially get all the Fashion_MNSIT testing and training dataset and split the training data into training and validation sets.The pixel value is in the range of 0-255 so we will normalise the dataset by dividing with 255.0. For easy application of the matmul function of the numpy with the weights for all the examples we transpose the matrix and return them.

Now we need to define the functions that is getting used in forward and back propogation. For forward propogation we need a non linear activation function. Sigmoid, Relu and tanh are the three main activations used and hence they are defined. Also since the last layer is a probability distribution we use a softmax function. Then for backpropogation we would also need the differentiation functions for this functions.

initParameters is the function that we use to initialize the weights and biases for the first forward propogation. But if we randomly assign large values then there is a chance that the values might grow to a very large value and overflow. That is the reason why we are multiplying the random values by a value of 0.01.

In forward_propogation the input is the X traiing matrix which will have a 784 row matrix where 784 is the values of all the pixels. The number of columns of the matrix depends on the size of the batch chosen. The output will be a probability distribution over the 10 classes(ie, The sum of the distribution should be 1)

For back_propogation, we first need to find out which loss function to use. For the probability distribution the better option would be cross entropy. But to observe the difference in Question 8, we cadd both cross-entropy as well as mean squared error loss.The back propogation will give the gradients of the weights and biases. This can be used to adjust the weights and biases depending upon the optimizer we are chosing.

Following Optimizers are defined and more can be defined as per requirement:
1. gd-Vanilla Gradient Descent
2. mgd-Momentum based Gradient Descent
3. ngd-Nesterov's Momentum based Gradient Descent
4. rmsprop- Root mean squared Propogation
5. adam-Adaptive Momentum based Gradient Descent
6. nadam-Neaterovs adaptive momentum based Gradient Descent
