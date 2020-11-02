# In the second exercise for this lab, we are going to build a convolutional 
# neural network (CNNs). As we talked about in class, CNNs are designed for
# cases where identifying the same pattern in different parts of the input
# is useful. Much of this exercise is similar to the first part. We'll
# explore different parameters in the CNN architecture to see what works 
# well for this dataset.

# PART 1: LOAD THE DATA

# Most of this is identical to what we did in the first part.

library(keras)

fashion_mnist <- dataset_fashion_mnist()

train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y

test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

train_images <- train_images / 255
test_images <- test_images / 255

# Here's the only change. With a CNN we need to preserve the spatial structure of the images
# as we input them into the network. Instead of flattening out the 28x28 pixel images into
# a single 784 element array, we will keep the 28x28 shape. We also need to make one small
# modification based on how Keras implements convolutional layers. Keras expects an image
# to have 3 dimensions: x, y, and color channel. Since we are using grayscale images our
# images are going to be 28 x 28 x 1. If we were using color images the images would be
# 28 x 28 x 3. The three would represent the three different color channels: red, green, 
# and blue.

train_images <- array_reshape(train_images, c(dim(train_images)[1], 28, 28, 1))
test_images <- array_reshape(test_images, c(dim(test_images)[1], 28, 28, 1))

# PART 2: CREATE THE MODEL

# We start the same way as before, by creating an empty sequential model.
model <- keras_model_sequential()

# Now we can add layers to out model.
# We'll use a few different kinds of layers to create the network:

# layer_conv_2d: Creates a convolutional layer. 
# layer_max_pooling_2d: Creates a max-pooling layer.
# layer_flatten: Takes a multi-dimensional structure, like 2D feature maps, and flattens it.
# layer_dense: A fully-connected layer.
# layer_dropout: Implements dropout to avoid overfitting. Will explain in class.

# Here's a basic CNN architecture. We'll talk about each of the steps.

model %>%
  layer_conv_2d(filters=64, kernel_size = c(4,4), activation='relu', input_shape=c(28,28,1)) %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dense(units=256, activation = 'relu') %>%
  layer_dense(units=10, activation='softmax')

# We can use the summary() function to see the structure of the network. 3 million+ parameters!
summary(model)

# Let's fit it. This is the same as before.
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs=10, validation_data = list(test_images, test_labels))

# Notice the loss is going really low, and the accuracy gets quite high. But the validation loss initially
# drops and then goes up, and the validation accuracy plateaus well below the training accuracy. What's going
# on? This is a clear example of over fitting. The model is learning patterns that are specialized for the
# training set. 

# Here's another model. Let's talk about how it is different from the first one.

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters=64, kernel_size=c(4,4), activation='relu', input_shape=c(28,28,1)) %>%
  layer_max_pooling_2d() %>%
  layer_conv_2d(filters=32, kernel_size = c(3,3), activation='relu') %>%
  layer_max_pooling_2d() %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>%
  layer_dense(units=256, activation = 'relu') %>%
  layer_dropout(0.25) %>%
  layer_dense(units=10, activation='softmax')

summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(train_images, train_labels, epochs=10, validation_data = list(test_images, test_labels))

# How does this model compare? The validation accuracy (ability to classify images that it wasn't trained on)
# is pretty similar, with values around 91%. But here we don't have the clear over fitting problems. And we
# did it with only 10% of the trainable parameters!

