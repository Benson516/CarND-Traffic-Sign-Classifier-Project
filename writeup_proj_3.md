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
[image201]: ./output_images/data_preprocessing/img_before.png "Original image"
[image202]: ./output_images/data_preprocessing/img_after.png "Image after normalizing"
[image3]: ./output_images/training_results/plot_image_train.png "Training result"
[image4]: ./output_images/test_image_proper_size_plot/test_1.png "Traffic Sign 1"
[image5]: ./output_images/test_image_proper_size_plot/test_2.png "Traffic Sign 2"
[image6]: ./output_images/test_image_proper_size_plot/test_4.png "Traffic Sign 3"
[image7]: ./output_images/test_image_proper_size_plot/test_5.png "Traffic Sign 4"
[image8]: ./output_images/test_image_proper_size_plot/test_7.png "Traffic Sign 5"
[image901]: ./output_images/test_prob_bar_chart/test_1.png "Traffic Sign 1"
[image902]: ./output_images/test_prob_bar_chart/test_2.png "Traffic Sign 2"
[image903]: ./output_images/test_prob_bar_chart/test_4.png "Traffic Sign 3"
[image904]: ./output_images/test_prob_bar_chart/test_5.png "Traffic Sign 4"
[image905]: ./output_images/test_prob_bar_chart/test_7.png "Traffic Sign 5"
[image10]: ./output_images/test_activation_viz/test_4.png "Traffic Sign 5"



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

Before sending data into network, I constructed preprocessing steps:
- normalizing intensitive of image

This is the only manually-coded proccess in sign classification. There is a prior knoledge I inserted that the traffic sign is not utilizing the intensity as a feature for distinguishing between signs; therefore, I normalize the intensity of each channel by the following mapping (x_min, x_max) --> (0, 255), where x_min/x_max is the minimum/maximum pixel value among all channels. However, I don't convert the original image into grayscale because I think that the color is an important feature in design of traffic signs.

The following is the code for normalizing the data. Note that I only choose the ROI of `image[5:27:, 5:27:]` to calculate the minimum and maximum pixel values because the sign is mostly located at the center region of image.

```python
def pre_processing(a_single_x):
    """
    This processing is for every single data.
    Note: I don't process all data at once because 
           there might have some adaptive function in preprocessing 
           which is different for different input data.
    """
    br = 5
    x_min, x_max = np.min(a_single_x[br:-br:]), np.max(a_single_x[br:-br:])
    a_single_x_modified = np.clip( (a_single_x.astype(np.float32) - x_min) * (255.0/(x_max-x_min)), 0.0, 255.0 ).astype(np.uint8)
    return a_single_x_modified
```


Here is an example of a traffic sign image before and after normalizing.

![alt text][image201]
![alt text][image202]

Fig. 4 Image before (top) and after normalizing (down)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description					| 
|:---------------------|--------------------------------:| 
| Input         		| 32x32x3 RGB image, `dtype=np.uint8` | 
| Casting         		| 32x32x3 RGB image, `dtype=np.float32` | 
| Convolution 1, 3x3   	| 1x1 stride, VALID padding, outputs 30x30x16 |
| Leaky-RELU 1			| leak-slope 0.2	    	|
| Max pooling 1	      	| 2x2 stride, VALID padding, outputs 15x15x16   |
| Convolution 2, 3x3    | 1x1 stride, VALID padding, outputs 13x13x30 |
| Leaky-RELU 2			| leak-slope 0.2					    	|
| Max pooling 2	      	| 2x2 stride, VALID padding, outputs 6x6x30 |
| Flaten	      	    | inputs 6x6x30, outputs 1080 	|
| Fully connected 1		| inputs 1080 , outputs 120        		|
| Leaky-RELU 3			| leak-slope 0.2					    	|
| Dropout 1| |
| Fully connected 2		| input 120 , output 84        	    	|
| Leaky-RELU 4			| leak-slope 0.2					    	|
| Dropout 2| |
| Fully connected 3		| input 84 , output 43        		    |
| Softmax				| 43 output	classes in one-hot coding  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For tranning the model, I use `AdamOptimizer` with default hyper parameters to optimize the model. Other hyper parameters are listed in the below table.

| Parameter      | value			| 
|:--------------|----------------:| 
| learning rate **High**, for accuracy in [0, 0.97)      | 0.0018    |
| learning rate **Mid**, for accuracy in [0.97, 0.995)    | 0.0008    |
| learning rate **Low**, for accuracy in [0.995, 1.0]     | 0.0005    | 
| batch size | 128 |
| number of epochs | 30 |
| keep_prob of dropout layers| 0.5|
| beta for regularization | 0.2 |

The loss function for trainning model is calculated as follow.
```python
loss_operation = tf.reduce_mean(cross_entropy + beta * regularizer)
```
, where the **regularizer** is calculated by the following tf operation.
```python
# weights['out'] is the weight matrix of "Fully connected 3" layer defined above
regularizer = tf.nn.l2_loss(weights['out'])
```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.961
* test set accuracy of 0.945

I chose `LeNet-5` as the initial architecture; however, after first time of training, the validation accuracy turns out to be less than 0.93 while the accuracy of training set is approaching to 1.0. I realize this is a symble of overfiting; therefore, I utilize the regulization on weights of last fully-connected layer and drop-out (p=0.5) after the activation layer of each fully-connected layer. This provide a good accuracy (0.93) on the validation set of the German Traffic Sign data set that satisfied the minimum criterion of the project. 

However, when I tested the model with images that are **not** from the German Traffic Sign Dataset, the model failed to classify any of the sign. After I visualize the activation of the first convolution layer using the `outputFeatureMap()` function, I discovered that the features it detected is quite restricted in the original LeNet-5 setting (6 channels for output featuremap). More specifically, at least one channel activated at the region of *sky* but none of them activated at the pattern drawn in the sign. 

Finally, I decided to increase the number of channel in first convolutional layer to 16. This resulted in higher accuracy in tranning set, validation set, and test set. However, the predictions on new images still quite low. The analysis will be described below.

The following plots are the losses and accuracies during each epoch of training.

![alt text][image3]

Fig. 5 Losses and accuracies during training

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic sign images I found on the web, which have be crorped and resized to proper size. All the signs I found here is "Children crossing (y=28)".

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Fig. 6 Five test images found on web 

Original images can be found by following links:
- https://www.alamy.com/a-german-traffic-sign-with-children-on-it-in-hanover-germany-03-july-image158680106.html
- https://www.123rf.com/photo_125616431_german-road-sign-children-crossing-way-to-school.html
- https://cn.depositphotos.com/2069245/stock-photo-priority-road-and-children-crossing.html
- https://www.alamy.com/children-crossing-warning-sign-in-germany-image3302585.html
- https://www.alamy.com/stock-photo-traffic-signs-children-crossing-and-speed-limit-defaced-sticker-covered-84970621.html


All the "Children crossing (y=28)" sign are hard to classify because of the folowing reasons.
- The pattern on the sign is blur in 32x32 image.
- There are several similar triangle-shaped sign in the dataset that are hard to identify in 32x32 resolution, even by human.
- There are 480 "Children crossing (y=28)" image in traning data, which are relatively rare in dataset. 

Beside the common reason of the difficulty to classify the "Children crossing (y=28)", there are some possible difficulties in each individual image.
- The first image might be difficult to classify because it's tilting to the right and the color saturation is low related to other image.
- The second image might be difficult to classify because it's tilting and contains other sign underneath the "Children crossing (y=28)".
- The third image is hard to classify because there is another sign above the "Children crossing (y=28)".
- The forth image is hard to classify because it's sheared and the background is relatively complex.
- The fifth image is hard to classift because there is another sign underneath the "Children crossing (y=28)", the color of sign is shifted, and the background is complex.






#### 2. Discuss the model's predictions on these new traffic signs and compare the results to prediction on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image	(Lable)		    |     Prediction	   | 
|:----------------------|:---------------------| 
| test_1.jpg (28)  		| 30 [Beware of ice/snow]           | 
| test_2.jpg (28)  		| 13 [Yield]                        | 
| test_4.jpg (28)  		| 29 [Bicycles crossing]            | 
| test_5.jpg (28)  		| 24 [Road narrows on the right]    | 
| test_7.jpg (28)  		| 28 [Children crossing]            | 


The model can only correctly identify one of the five "Children crossing" signs, which gives an accuracy of 20%. The recall rate for "Children crossing" according to this test result is calculated as `1/5=20%`, and the precision for "Children crossing" is calculated as `1/1=100%` (Since there is no other class in this test set, it must be 100% if TP>=1). The result is far worse than the accuracy for test-set form original German Traffic Sign Dataset.

From human perspective, the reason for the classifier to be confused may be
| Image	(Lable)		    |     Prediction	   | Possible Confusion Reason |
|:----------------------|:---------------------|:---| 
| test_1.jpg (28)  		| 30 [Beware of ice/snow]           | The sign is in triangle shape with red boarder. |
| test_2.jpg (28)  		| 13 [Yield]                        | The sign is in (reversed) triangle shape with red boarder. |
| test_4.jpg (28)  		| 29 [Bicycles crossing]            | The sign is in triangle shape with red boarder. |
| test_5.jpg (28)  		| 24 [Road narrows on the right]    | The sign is in triangle shape with red boarder. |
| test_7.jpg (28)  		| 28 [Children crossing]                    | (correct) |


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell `#35` of the Ipython notebook.

For the first image, the model is confusing around the "Beware of ice/snow" sign (probability of 0.731) and the "Road work" sign (probability of 0.227), which are both in red triangle shape. The top five soft max probabilities are shown in the below image.

![alt text][image901]

Fig. 7 Probabilities output of softmax layer for "test_1.jpg"

For the second image, the model is quite shure aboud there is a "Yeild" sign in the image (probability of 0.832) while confusing with the "Priority road" sign. This is an interest result where the image indeed contained a "Priority road" sign at the top. The top five soft max probabilities are shown in the below image.

![alt text][image902]

Fig. 8 Probabilities output of softmax layer for "test_2.jpg"

For the third image, the model is confusing around the "Bicycles crossing" sign (probability of 0.929) and the correct answer (probability of 0.070), which are both in red triangle shape. The top five soft max probabilities are shown in the below image.

![alt text][image903]

Fig. 9 Probabilities output of softmax layer for "test_4.jpg"

For the forth image, the model is confusing around the "Road narrows on the right" sign (probability of 0.697) and the correct answer (probability of 0.099), which are both in red triangle shape. The top five soft max probabilities are shown in the below image.

![alt text][image904]

Fig. 10 Probabilities output of softmax layer for "test_5.jpg"

For the fifth image, the model is has correct answer (probability of 0.840). The top five soft max probabilities are shown in the below image.

![alt text][image905]

Fig. 11 Probabilities output of softmax layer for "test_7.jpg"

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


The following figure is the activation of first convolutionl layer with "test_4.jpg" as input image.

![alt text][image10]

Fig. 12 Visualization of the activation on first convolusion layer at "test_4.png"


From the figure we can found the following insight about network
- Channel 0, 7, 10  not trust worthy because they largly activated at sky.
- Channel 2, 5, 11, 15 are quite sensitive to the red triangle boarder.
- Channel 1, 6, 9, 11 get different part of the pattern in the sign.