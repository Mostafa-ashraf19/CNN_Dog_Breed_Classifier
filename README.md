# Dog Breed Classifier Using CNN 

 - A [canine’s breed estimator](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/Report/CNN-Dog%20Breed%20Classifier-Project%20Proposal.pdf) exploring CNN principles for classifications. A model will be training with real-world data then after final training, the model will be supplied with dog images and predict its breed. If a model is supplied with a human image, the model will estimate a closet similarity for a dog breed.

## Problem Statement

 - A canine’s breed as we discussed before, its used to classify breeds of dogs, and if it’s supplied with a human image its give me closet similarity for a human face to an estimated breed dog, but let’s talk more, we start using a cascade classifier provide from OpenCV it’s a face detector pre-trained model it’s trying to detect a human face and try to find it’s accuracy on a chunk from our data humans and dogs it’s given me high accuracy in human images, we start using a VGG-16 model in PyTorch it’s trained on an ImageNet dataset it contains around 1000 categories we used it’s to classify an image it’s a dog or not after making some data preprocessing e.g. resize, cropping, normalization, etc... then also find it’s accuracy and its give us a high accuracy on dogs images. Finally, we start to developed our CNN model from scratch using PyTorch and train - validate, and test this model. we also apply a concept of transfer learning to make a prediction. Finally, all models mentioned make some preprocessing for data.

## Metrics
 - According to the precision and recall concept when we used in binary class  classification, will apply this concept into multiclass classification,  a typical multi-class classification problem, we need to categorize each sample into 1 of N different classes,
Similar to a binary case, we can define precision and recall for each of the classes. 
So, Accuracy = correct / total size
Or  = true positive + true negative / dataset size 

## Data Exploration and Visualization
 - Dataset I’ve used consists of 13233 [human images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) and 8351 [dog images](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), its supplied to CNN model I have created in Project, also in a transfer learning section I’m used a VGG16 Model and this is trained on ImageNet, it is a very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories.  
 ![dogsamples](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/dog%20breed%20freq%20vis.png?raw=true)
 
 ## Algorithms and Techniques
  - The classifier I have used follows the principle of convolutional neural networks which is the state-of-the-art algorithm for most image processing tasks, including classification. It needs a large amount of training data compared to other approaches, its division into three main approaches, Pretranided haar cascade classifier model for face detection, VGG-16 pre-trained model for dog detector or it’s breed, CNN written from scratch, so let’s discuss CNN architecture. 
![CNNarch](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/CNN.jpeg?raw=true)

## Data Preprocessing
 - The preprocessing done in the “Prepare data” notebook consists of the following steps: 
     1. The list of images is randomized.
     2. The images are divided into a training set and a validation set.
     3. Make some transformation into images cropping.
     4. Normalized images and convert them into tensors.
     5. Pass images to the data loader with batch size equal thirty then shuffle it.
     ![imageT](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/image%20transforming.png?raw=true)
##  Implementation
 - A ConvNet is able to successfully capture the Spatial and Temporal dependencies in an image through the application of relevant filters. The architecture performs a better fitting to the image dataset due to the reduction in the number of parameters involved and the reusability of weights. In other words, the network can be trained to understand the sophistication of the image better. The role of the ConvNet is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction. It’s consists of a convolutional it has a kernel n * m dim with stride(step of movement), also it has a pooling layer and when a stride is higher than 2 it’s called downsample pooling can be max pooling or average pooling, etc…, 

![MyCNN](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/MyCNN.png?raw=true)

## Free-Form Visualization
 1.  Face detector and dog detector examples
 
 
  ![Hfd](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/face%20detection.png?raw=true)
  
  
  ![Dpred](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/dog%20prediction.png?raw=true)
  
 2. CNN breed classifier results 
 
 
 ![Hres](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/results/human_res1.png?raw=true)
 
 
 ![Dres](https://github.com/Mostafa-ashraf19/CNN_Dog_Breed_Classifier/blob/master/assets/results/dog_res1.png?raw=true)
 
