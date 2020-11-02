# For part 3 of the lab, we are going to do a simplified replication of
# the Lake, Zaremba, Fergus, & Gureckis (2015) experiment.

# This will demonstrate how to use pre-trained networks as part of a model.

# PART 1: LOADING THE PRE-TRAINED MODEL

# We start by loading keras...

library(keras)

# For our model, we are going to use one of the convnets that is built into
# keras. This model, VGG16, is a convolutional neural network that has already
# been trained on the imagenet dataset. Imagenet consists of objects from 1,000
# different categories.

# To load the model we call the appropriate application_ function. If you 
# want to see other models that keras has built in, you can type
# keras::application_ and the autocomplete should give you many choices.

# There's no particular reason that we are using VGG16 here. It's just an 
# example of a convnet, and it's closest to the kinds of nets that were
# tested in the paper. Some of the other nets are more sophisticated.

model <- application_vgg16()

# We can use the summary() command to see the structure of this network:

summary(model)

# Notice that it uses the same kinds of layers that we used to build our own
# convolutional neural network. Just more of them. It also has WAY more
# parameters -- more than 138 million. Good thing we don't need to train it!

# PART 2: THE EXPERIMENT - DATA GATHERING

# In the paper, Lake et al. got human typicality ratings for a bunch of images in 
# several different categories. We don't have that luxury, so we'll conduct 
# a different kind of human data analysis. 

# Here's a list of the 1,000 different categories that imagenet is trained on:

# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# Pick 8 categories from the list, and use an image search (like Google Image Search)
# to find two photos from the category. One photo should be an image that you
# think is highly typical of the category. The other should be an image that you 
# think is atypical. The atypical image should still clearly be a member of the category.

# Save the image files on your computer, naming them something sensible, like
# category-name-typical.jpg and category-name-atypical.jpg.

# Once you've got 16 images, upload them to the GPU server by clicking the Files tab
# to the right and clicking the upload button. You can upload them one by one or you 
# can create a ZIP archive and upload them as a group.


# PART 3: THE EXPERIMENT - RUNNING THE MODEL

# To run these images through the model, we need to do a little bit of preprocessing.
# The model was trained on images that were 224 x 224 pixels. We need to resize the images
# you found to match this size. This might stretch the images in odd ways, but convnets are
# pretty robust to this kind of image manipulation. 

# Load the image, specifying the target dimensions. Pick one of your 16 images to start.
# Then you'll come back and run this section of the script again for the remaining images.
image <- image_load('images/pretzel-atypical.jpg', target_size = c(224,224))

# Next, we need to take the image and convert it into an array. The dimensions of the array
# will be 224 x 224 x 3. The third dimension (with three values) indexes the colors of the 
# image in RGB space. The first channel is the intensity of the red channel, the second
# the green channel, and the last is the blue channel.
image.arr <- image_to_array(image, data_format = "channels_last")

# Next, we have to reshape this array to add a single dimension at the front. This dimension
# is normally where we would store multiple samples. Recall that in the fashion MNIST 
# data set, our input images were 60000 x 28 x 28. The 60000 were the 60000 different images.
# Here we only have one image at a time, so we just reshape this to have a single example.
image.arr.reshaped <- array_reshape(image.arr, c(1,224,224,3))

# Finally, we use a preprocessing function that ensures our image has the same kinds of 
# representations of pixel intensity as the original training data. 
image.arr.reshaped.processed <- imagenet_preprocess_input(image.arr.reshaped, mode="tf")

# We're finally ready to run the model. This will be really fast, because all we need to
# do is a single forward-pass through the model. Even for models with a hundred million
# parameters, that's a quick operation. (The expensive part is training the model, but 
# this network is already pretrained.) To run the model we use the predict() function, 
# passing in the array of preprocessed image data.
prediction <- model %>% predict(image.arr.reshaped.processed)

# The output of the this step is the activity of the final layer of the network:
# 1,000 units with the softmax activation function. Because it is a softmax layer,
# The activity of the 1,000 units adds up to 1, and the value of each unit is the model's
# probability that the image belongs to the corresponding class.

# keras has a helper function to convert this 1,000 element array into a more useful representation
# of which classes the model thinks the image belongs to. 

imagenet_decode_predictions(prediction, top=5)

# Hopefully the correct class is one of the top 5. If not, you can increase the value of top to see more
# predictions.

# Record the score (probability) for the target category, and whether the image was typical or atypical,
# in this spreadsheet:

# https://docs.google.com/spreadsheets/d/1ZcxSLvrrrb_cs_aAqUNlY3QQgS9C28hYROiQOPDtF4U/edit#gid=0

# Now repeat the analysis for the remaining images, and add them to our dataset.

# BONUS: AN ANALYSIS OF OUR CLASS DATA

# Using the googlesheets4 package we can pull data in from the live spreadsheet!
# You'll need to install the googlesheets4 package to do this.

library(googlesheets4)
library(dplyr)
library(ggplot2)

# This command tells googlesheets that we aren't authenticating, since the sheet is public to view.
gs4_deauth()

# read in the data from the sheet
our.data <- read_sheet('https://docs.google.com/spreadsheets/d/1ZcxSLvrrrb_cs_aAqUNlY3QQgS9C28hYROiQOPDtF4U/edit?usp=sharing')

# plot the data by typicality
ggplot(our.data, aes(x=Typicality, y=Score)) +
  geom_point(position = position_jitter(width=0.1))+
  theme_bw()

# We can use a t-test to see if the items we judged as more typical are 
# classified more confidently by the network. We'll use a paired-samples t-test
# because we are comparing within each category-experimenter combination.

t.test(Score ~ Typicality, data=our.data, paired=T)







