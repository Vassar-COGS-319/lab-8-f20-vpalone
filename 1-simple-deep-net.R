Sys.setenv('CUDA_VISIBLE_DEVICES'="0")

# The aim of this lab exercise is to show you how to construct and train a deep
# neural network using modern machine learning software packages.

# We are going to use tensorflow and keras.

# tensorflow is a machine learning library developed by Google. It is open source,
# widely used, has lots of instructional content available online, and works with
# R. keras is an abstraction layer on top of tensorflow. It simplifies the process
# of creating and training deep neural networks by making the language more
# like the kind of language we use to talk about neural networks conceptually.

# PART 1: LOADING IN A DATASET

# Training any deep neural network from scratch requires a large set of data. 
# For this exercise we are going to use a data set called Fashion MNIST.
# This data set has 70,000 grayscale images of 10 different categories of clothing:
# 0 = T-shirt/top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle boot

# Each image in the dataset is 28x28 pixels (784 pixels total).

# Fashion MNIST is meant to be a kind of "benchmark" test. It's not too hard to 
# create a neural network that can classify the images correctly, but it's also
# not trivial. Showing that a network can work on Fashion MNIST is a way to
# demonstrate feasibility. Because it is a commonly-used benchmark, keras includes
# a function to easily get it.

library(keras) # this is installed on our GPU-server

fashion_mnist <- dataset_fashion_mnist()

# If you look at fashion_mnist in your environment panel, you'll see that it is a 
# list with two elements: train and test. The train list contains data for 60,000
# images, used to train the network, and the test list contains data for 10,000
# images, used to test the network.

# Why do you think we need to test the network with different images than the ones
# that we train the network with? What problem does this relate to from our
# previous discussions of model fitting?


# Let's create four variables here to simplify access to the data:

train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y

test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

# We can look at the raw data associated with one image like this:

train_images[1,,]

# It's a 28 x 28 matrix, with values from 0-255. 0 = black; 255 = white.

# It's going to be a lot easier to work with the data if we change the
# scale from 0-1 instead of 0-255. Let's do that for both the training
# and test images.

train_images <- train_images / 255
test_images <- test_images / 255

# Here's a function to visualize one image:

visualize.image <- function(img){
  img <- t(apply(img,2,rev)) # rotates the image clockwise 90 degrees
  image(1:28,1:28, img,col=gray((0:255)/255), xaxt = 'n', yaxt ='n',xlab=NA, ylab=NA)
}

# Let's try it on the first image:

visualize.image(train_images[1,,])

# What is it? We can look at the category label for each image by looking
# at the corresponding label.

train_labels[1]

# This is category 9. If you scroll up you'll see that category 9 is "ankle boot".

# PART 2: BUILD THE NEURAL NETWORK!

# The real power of Keras is that we can assemble a set of layers to construct different
# neural network architectures. Here we are going to build a simple dense network where
# each layer is fully connected to each other layer.

# Initialize the model

model <- keras_model_sequential()

# Add layers

model %>%
  layer_flatten(input_shape=c(28,28)) %>%
  layer_dense(units=128, activation = 'relu') %>%
  layer_dense(units=64, activation = 'relu') %>%
  layer_dense(units=10, activation = 'softmax')

# The first layer (layer_flatten) simply takes our 28 x 28 images and flattens them out
# into a 784 element vector. There are no trainable weights associated with this layer.
# In fact, we could do this flattening ourselves outside of the neural network, but
# this layer makes it so easy that it would be silly to not do it here.

# The second layer has 128 units. Each unit is fully connected to all 784 inputs.
# The activation function for this unit is relu.

# The final layer has 10 units. Each unit is fully connected to all 128 units from the
# previous layer. These 10 output units can be used to represent each of the 10 categories
# that the network is learning. Our goal will be that every time an ANKLE BOOT is shown
# as input, the activation of UNIT 9 will be 1.0 and the activation of the other units
# will be 0. The softmax activation function ensures that the total activation of the
# layer adds up to 1.0. This allows us to treat the output as a probability distribution.

# PART 3: COMPILE THE MODEL

# Once we have the network architecture in place we need to specify a few more features
# before we are ready to train the model. 

# We need to specify the discrepancy function.
# This is what we will use to compare the desired and actual output. Keras has many different
# choices, each appropriate for a different kind of data. Here we use 
# 'sparse_categorical_crossentropy', which is an appropriate discrepancy function when
# you are classifying each item into one of many categories and the output represents
# the probability of being in a particular category. Note that in machine learning lingo
# the discrepancy function is called the loss function.

# We also need to specify the optimizer.
# The optimizer is the algorithm used to update the weights of the neural network.
# It's analogous to choosing Nelder-Mead vs. Genetic Algorithms for parameter estimation.
# There are several different optimizers available, and for more complicated models
# it might be worth trying different optimizers out to find the one the works best.
# Here we can just pick one and go with it.

# Finally, we can specify metrics that we want keras to track while the network is training.
# This will help us understand if the network is learning to categorize the objects correctly.

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# PART 4: TRAIN THE MODEL!

# To train the model we provide it with the training input, the output we want it to produce,
# and the length of training. In machine learning lingo, an epoch is one presentation of all
# the inputs to the network. Here we will train the network for 10 epochs. This will mean 
# the network sees each of the 60,000 images in the training set 10 times.

model %>% fit(train_images, train_labels, epochs=10, validation_data = list(test_images, test_labels))

# PART 5: CHECK THE MODEL

# We can use the predict() function to see how the model classifies the images
# in the test data.

test.predictions <- model %>% predict_classes(test_images)

# We can compare these predictions with the test_labels to see where the model 
# gets the right classification

wrong.answers <- which(test.predictions != test_labels)

# We can visualize a few of these to see what the network had trouble with.

visualize.image(test_images[wrong.answers[1],,]) # show the image
test.predictions[wrong.answers[1]] # category that the model predicted
test_labels[wrong.answers[1]] # correct answer





