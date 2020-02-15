# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image101]: ./output_images/data_statistic/distribution_train.png "Distribution of tranning set"
[image102]: ./output_images/data_statistic/distribution_valid.png "Distribution of validation set"
[image103]: ./output_images/data_statistic/distribution_test.png "Distribution of test set"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I accesed to the numpy shape member to get imformation of the traffic signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

Note:

I can use `len(set(y_train))` command to calculate the unique number of classes **in** data set. However, it is possible that the dataset does not cover all the classes we defined. I use `len(signnames_dict)` instead. Where `signnames_dict` is a python dictionary that stores the mapping of `(class number):(class definition string)` listed in `signnames.csv` files.




#### 2. Include an exploratory visualization of the dataset.

The dataset has totally `43` classes, and their distributions are shown in Fig.1, 2, and 3. From the plot we can see that the distribution is not quite even; hoever, the distribution is similar among three data set.

![alt text][image101]
Fig. 1 The distribution of trainning set.

![alt text][image102]
Fig. 2 The distribution of trainning set.

![alt text][image103]
Fig. 3 The distribution of trainning set.



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image, `dtype=np.uint8` | 
| Casting         		| 32x32x3 RGB image, `dtype=np.float32` | 
| Convolution 1, 3x3   	| 1x1 stride, same padding, outputs 32x32x16 |
| Leaky-RELU 1			| leak-slope 0.2	    	|
| Max pooling 1	      	| 2x2 stride,  outputs 16x16x16	|
| Convolution 2, 5x5    | 1x1 stride, same padding, outputs 16x16x16  |
| Leaky-RELU 2			| leak-slope 0.2					    	|
| Max pooling 2	      	| 2x2 stride,  outputs 8x8x16	|
| Flaten	      	    | inputs 8x8x16, outputs 1024 	|
| Fully connected 1		| inputs 1024 , outputs 120        		|
| Leaky-RELU 3			| leak-slope 0.2					    	|
| Dropout 1| |
| Fully connected 2		| input 120 , output 84        	    	|
| Leaky-RELU 4			| leak-slope 0.2					    	|
| Dropout 2| |
| Fully connected 3		| input 84 , output 43        		    |
| Softmax				| 43 output	classes in one-hot coding  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For tranning the model, I use `AdamOptimizer` with default hyper parameters to optimize the model. Other parameters are listed in the below table.

| Parameter      | value			| 
|:--------------:|:----------------:| 
| learning rate **High**, for accuracy in [0, 0.9)      | 0.0018    |
| learning rate **Mid**, for accuracy in [0.9, 0.99)    | 0.0008    |
| learning rate **Low**, for accuracy in [0.9, 1.0]     | 0.0005    | 
| batch size | 128 |
| number of epochs | 30 |
| keep_prob of dropout layers| 0.5|
|beta for regularization | 0.2 |

The loss function for trainning model is calculated as following
```python
loss_operation = tf.reduce_mean(cross_entropy + beta * regularizer)
```
, where the **regularizer** is calculated by the following
```python
# weights['out'] is the weight matrix of "Fully connected 3" layer defined above
regularizer = tf.nn.l2_loss(weights['out'])
```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


