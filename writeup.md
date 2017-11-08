# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/plot.png "Visualization"
[image2]: ./images/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/sign1.png "Traffic Sign 1"
[image5]: ./images/sign2.png "Traffic Sign 2"
[image6]: ./images/sign3.png "Traffic Sign 3"
[image7]: ./images/sign4.png "Traffic Sign 4"
[image8]: ./images/sign5.png "Traffic Sign 5"
[image9]: ./images/expanded0.png "Traffic Sign Rotated Right"
[image10]: ./images/expanded1.png "Traffic Sign Rotated Left"
[image11]: ./images/expanded2.png "Traffic Sign Scaled"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/julson/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I plotted the training, validation and tests sets on a histogram to see the distribution of traffic signs across all the 43 classes. The distribution is a bit uneven, so it might be useful
to generate fake data other classes to balance it out.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For pre-processing, I normalized the images to reduce the oscillations when the learning rate gets applied and turn them into grayscale in order to reduce the dimensions of the input (they're going to get reduced as they pass through the network anyway)

![alt text][image2]

I expanded the data set to add 3 more images per sign, with the images rotated 10 degrees to the right and left, and with one scaled up to +10% of its size. This was intended to make it more robust to potential variations of input and to reduce the number of epochs needed to achieve convergence (although this expectedly made each epoch run slower)

![alt text][image9] ![alt text][image10] ![alt text][image11]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   			  		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120 									|
| RELU                  |                                               |
| Dropout               | keep_prob @ 0.5                               |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Dropout               | keep_prob @ 0.5                               |
| Fully connected       | outputs 43                                    |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Not much of the training pipeline was changed from the LeNet Lab code (if it ain't broke...). The classes are one-hot encoded for easier classification (instead of dealing with probabilities) before passing it through TensorFlow's soft_max_cross_entropy_with_logits. The AdamOptimizer was used to minimize the loss for backpropagation, which I think was beneficial given its ability to adjust the learning rates per parameter.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.9%
* test set accuracy of 94.6%

The LeNet architecture from the lab was chosen as it's the architecture I'm most familiar with. Despite its age (and my laziness), it's still a very relevant architecture due to its speed, simplicity and its ability to classify images (given that it was used for character recognition).

Using just the vanilla architecture taken from the lab code, it produced a validation set accuracy to around ~89%. Dropout was then added between the 3 fully-connected layers to introduce some redundancy, which increased the accuracy to ~93%. Expanding the training set to include some scaled and rotated images finally pushed the final validation accuracy to ~96%.

Given that the test set is a completely new set of images and not a set that the network has 'seen' already, running it through the model and producing 94.6% provides some confidence that the model works (though it could be better).


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

These images were picked from a German traffic sign dataset (TestIJCNN2013.zip, with 300 images) that I've found on the web, which I think might be from the set from what the model was trained on, although the images were in its original size and had to be scaled and cropped to 32x32 by hand, which would hopefully introduce some variation if that were indeed the case.

The first image is pretty dark and slightly rotated to the left, which I felt like would cause some issues after converting the image to grayscale.

The second image is also rotated to the left. I haven't looked at any of the images in the training set, but I was assuming that everything was all properly centered, so I thought that this would cause some issues.

The third image is pretty straight-forward, although it's slightly aligned to the right, I thought that the CNN's translation invariance would cause no issues with this.

The fourth image is scaled to almost fill the entire image. I thought it would be one of the easiest to classify, but it's the one that had the most difficulty.

The last image would be difficult given that its dark, small and slightly off-center.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General Caution 		| General Caution      							|
| General Caution     	| General Caution                               |
| Right-of-way			| Right-of-way			  						|
| 50 km/h	      		| 30 km/h	     				 				|
| Road Narrows Right	| Road Narrows Right      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%, which I think was pretty accurate given that its constantly only having trouble with the 50 km/h sign, after multiple runs of the notebook.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the bottom of the "Step 4: Test a Model on New Images" section in the Jupyter notebook.

The first "General Caution" image was successfully classified with a 97.629% certainty.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .97629         		| General Caution								|
| .02348     			| Traffic Signals 	   							|
| .00021				| Pedestrians	   								|
| .000002	      		| Road Work	         		  	 				|
| .0				    | Dangerous curve to the right      	   		|

The second "General Caution" image performed even better, obtaining a 100% probability.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0           		| General Caution								|
| .0     	    		| Road Work      	   							|
| .0	    			| Traffic Signals  								|
| .0	         		| Pedestrians	         		  	 		    |
| .0				    | Dangerous curve to the right      	   		|

The third image, a "Right-of-way at the next intersection" sign, was also predicted successfully.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96032         		| Right-of-way at the next intersection		   	|
| .03799     			| Beware of ice/snow  							|
| .0119	    			| Pedestrians	   								|
| .0004 	      		| Dangerous Curve to the left 	 				|
| .0				    | Double curve                       	   		|

This is where it all falls apart, the model was pretty sure that it's a 30km/h sign, with a 99.371% certainty, where in fact that it is a 50km/h sign. The general similarity of the signs, coupled with a low res input might have blurred the lines a bit, but I'm not too sure.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99371         		| Speed limit (30km/h)              		   	|
| .00628     			| Speed limit (50km/h) 							|
| .00001    			| Speed limit (70km/h) 	   						|
| .0     	      		| Speed limit (80km/h)      	 				|
| .0				    | Speed limit (100km/h                	   		|

Lastly, for the "Road narrows on the right" sign, the model predicted it correctly, albeit a lot more uneasily compared to the previous successful guesses. It got it right with a 55.249% probability, although it's also seeing it as a "Bumpy road".

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .55249         		| Road narrows on the right            		   	|
| .25517     			| Bumpy Road 	        						|
| .08229    			| Traffic Signals    	   						|
| .07036           		| Bicycles Crossing         	 				|
| .03006	  		    | Wild animals crossing                	   		|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
