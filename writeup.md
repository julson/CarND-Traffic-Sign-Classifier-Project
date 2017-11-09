# **Traffic Sign Recognition**

---

This is the writeup for Traffic Sign Recognition Project from Udacity's Self-Driving Car Nanodegree. The link to the project code is [here](https://github.com/julson/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

### Data Set Summary & Exploration

We used the traffic signs from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) which was already provided in pickled format.

Using the pandas library to calculate summary statistics of the traffic
signs data set, I got:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Looking at the training, validation and test sets plotted on a histogram, it shows that the distribution is a bit uneven across the 43 classes, but the sizes are sufficient enough to train the classifier.

![alt text][image1]

### Creating the Architecture

Preprocessing the images is necessary to reduce the variations between images. Converting the images to grayscale reduces the dimensions of the input and takes out color from the equation. The images are then normalized to reduce the oscillations when the learning rate gets applied. Applying both produces training image of something like this:

![alt text][image2]

I expanded the data set to add 3 more images per sign, with the images rotated 10 degrees to the right and left, and with one scaled up to +10% of its size. This was intended to make it more robust to potential variations of input and to reduce the number of epochs needed to achieve convergence (although this expectedly made each epoch run slower)

![alt text][image9] ![alt text][image10] ![alt text][image11]

The layers of the neural net were based from the [LeNet Architecture](http://yann.lecun.com/exdb/lenet/), introduced in 1998 and was mainly used then to perform character recognition. It is also the architecture featured in the nanodegree's lab (which pretty much means that it's the only architecture I know for now). There have been a lot more neural network architectures developed since then, but it's still very relevant given it's simplicity, speed and its success in image classification.

There weren't a lot of changes to the architecture (if it ain't broke...), but adding dropout dramatically improved the prediction results. The table below shows all the layers, outputs and settings used:


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

### Training and Results

Not much of the training pipeline was changed from the LeNet Lab code. The classes are one-hot encoded for easier classification (instead of dealing with probabilities) before passing it through TensorFlow's soft_max_cross_entropy_with_logits. The AdamOptimizer was used to minimize the loss for backpropagation, which I think was beneficial given its ability to adjust the learning rates per parameter. The learning rate remained unchanged at 0.001.

Using just the vanilla architecture taken from the lab code, it produced a validation set accuracy to around ~89%. Dropout was then added between the 3 fully-connected layers to introduce some redundancy, which increased the accuracy to ~93%. Expanding the training set to include some scaled and rotated images finally pushed the final validation accuracy to 95.9%.


Finally, running the model through the test set yielded a 94.6% accuracy, which is pretty good given that the test set is a completely new set of images and not a set that the network has 'seen' already, proving that the model works.

### Introducing a New Set of Images

Next was to search 5 new German traffic signs from the web. The new images were picked from another German traffic sign dataset (TestIJCNN2013.zip, with 300 images), which is eerily from the same site where the training dataset was taken from. What I got were images that are in their original size and form, so I had to scale and crop them to 32x32 by hand, which hopefully introduced some variations to the input if that were indeed the case.

| Image         	|     Description	        					|
|:------------------|:---------------------------------------------:|
|![alt text][image4]|The first image is pretty dark and slightly rotated to the left, which I felt like would cause some issues after converting the image to grayscale.|
|![alt text][image5]|The second image is also rotated to the left. I haven't looked at any of the images in the training set, but I was assuming that everything was all properly centered, so I thought that this would cause some issues.|
|![alt text][image6]|The third image is pretty straight-forward, although it's slightly aligned to the right, I thought that the CNN's translation invariance would cause no issues with this.|
|![alt text][image7]|The fourth image is scaled to almost fill the entire image. I thought it would be one of the easiest to classify, but it's the one that had the most difficulty.|
|![alt text][image8]|The last image would be difficult given that it's dark, small and slightly off-center.|

Running these through the model gave me:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General Caution 		| General Caution      							|
| General Caution     	| General Caution                               |
| Right-of-way			| Right-of-way			  						|
| 50 km/h	      		| 30 km/h	     				 				|
| Road Narrows Right	| Road Narrows Right      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%, which I think was pretty accurate given that its constantly only having trouble with the 50 km/h sign, after multiple runs of the notebook.

### Results in Detail

Digging through the probabilities generated by the model, I gathered the top five results for each image. Here's what I got:

The first image was a "General Caution" sign, which was successfully classified with a 97.629% certainty.

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
