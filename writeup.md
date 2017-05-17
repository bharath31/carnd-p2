# **Udacity CarND Project 2- Traffic Sign Recognition**
### By Bharath Lalgudi Natarajan


The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/n_classes.jpg "Class distribution bar chart"
[image2]: ./writeup_images/dataset_example.png "Dataset Example"
[image3]: ./writeup_images/original_preprocessing.png "Normalization Before"
[image4]: ./writeup_images/normalized_preprocessing.png "Normalization After"
[image5]: ./writeup_images/download1.png "Traffic Sign 1"
[image6]: ./writeup_images/download2.png "Traffic Sign 2"
[image7]: ./writeup_images/download3.png "Traffic Sign 3"
[image8]: ./writeup_images/download4.png "Traffic Sign 4"
[image9]: ./writeup_images/download5.png "Traffic Sign 5"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the iPython notebook.  

Pandas library is used to calculate statistics of the traffic signs.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3rd code cell of the iPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the classes and a random example image from the dataset.

![alt text][image1]![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the iPython notebook.

I decided to normalize the image data to make darker looking images, like ones taken at night, a little more clear.

Here is an example of a traffic sign image before and after normalization.

![alt text][image3]![alt text][image4]

I experimented with converting to greyscale, but the results I was getting from my classifier didn't seem to change, so I removed it.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the iPython notebook. They were already split into these sets when downloaded.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth and sixth cells of the iPython notebook.

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max Pooling           | kernel size 2x2, outputs 5x5x16               |
| Flatten               | outputs 864                                   |
| Fully Connected       | outputs 120                                   |
| RELU                  |                                               |
| Fully Connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully Connected       | outputs 43                                    |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper-parameters such as learning rate.

The code for training the model is located in the seventh cell of the iPython notebook.

To train the model, I used softmax functions which calculates the probabilities of the traffic sign class based on the output by the last layer. The traffic sign class is encoded by a one-hot encoding.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventh and eighth cell of the iPython notebook.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 93.0%
* test set accuracy of 91.7%

I used an iterative approach to come up with my model.

My architecture started with the LeNet architecture from the LeNet lab, and I modified it using an iterative approach until I got to 93% accuracy on the validation set.

The initial architecture would not go beyond 86% accuracy on the validation set, so needed some modification. I attempted preprocessing the images with both normalization, (and grayscale, which I removed as discussed before). This was a marginal improvement but only by 1 or 2%. I made layers extract more features to possibly glean more information from the images, changed number of epochs, changed kernel sizes to be smaller and find smaller details, slowed down the learning rate, increased batch size (in attempt to speed things up a little bit), which ended up resulting in overfitting.

Then, I added more layers with smaller kernel sizes (3x3 instead of 5x5). I was finally able to achieve 93% accuracy with this. I also decreased the learning rate to 0.001 and increased the number of epochs to 20. This meant my classifier could find smaller details of the traffic signs several times over, and would learn slowly on more batches. In the end my classifier was still overfitting quite a bit, as seen with the training set accuracy.

Further improvements would be to combine large and small kernel sizes for my convolution layer, and use both of those outputs to give my classifier a better idea of large and small features of the traffic signs. Varying the dataset and using dropouts would help train the classifier by making it more difficult to overfit because of adjusted or randomly missing data.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9]

The first image might be difficult to classify because the symbols in its center aren't so clear when scaled down to a 32x32 image and there are not many examples of this sign in the test set compared to others.

The second image might be hard to classify because there are so many other speed signs like it.

The third image might be hard to classify because its color is a bit yellowed, but is otherwise a very straightforward image.

The fourth image might be hard to classify because it is at an angle and has part of another sign in it.

The fifth image might be hard to classify because it is mostly a solid block of red color, which can look similar to a stop sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the IPython notebook.

Here are the results of the prediction:

| Image                             | Prediction                       |
|:---------------------------------:|:--------------------------------:|
| Children crossing                 | No passing                       |
| Speed limit (30km/h)              | Speed limit (30km/h)             |
| Right-of-way next intersection    | Right-of-way next intersection   |
| Speed limit (30km/h)              | Speed limit (100km/h)            |
| No entry                          | Stop                             |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is far less than that of the validation or test set accuracy, a total of 53% less than that of the validation set. There are several reasons for this, including this being a much smaller data set and other reasons discussed in the next section.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the eleventh cell of the iPython notebook.

##### Image 1

For the first image, the model did poorly. Children crossing wasn't on its list of top 5 softmaxs. This could be in part because there weren't enough examples of this sign in the dataset.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .94                   | No passing                                    |
| .02                   | Speed limit (60km/h)                          |
| .02                   | End of no passing                             |
| .02                   | Keep right                                    |
| .00                   | Roundabout mandatory                          |

##### Image 2

My classifier predicted correctly for this one with near perfect certainty. The other signs, which had negligible probabilities, also happened to be other speed limit signs which look very similar.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Speed limit (30km/h)                          |
| .00                   | Speed limit (20km/h)                          |
| .00                   | End of speed limit (80km/h)                   |
| .00                   | Speed limit (70km/h)                          |
| .00                   | Speed limit (80km/h)                          |

##### Image 3

My classifier predicted correctly for this one with near perfect certainty.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Right-of-way next intersection                |
| .00                   | Beware of ice/snow                            |
| .00                   | Double curve                                  |
| .00                   | General caution                               |
| .00                   | Beware of ice/snow                            |

##### Image 4

My classifier predicted this incorrectly, but the second highest softmax was the correct label, and all the softmaxes were speed limit signs, which all look very similar. This image also contained parts of another sign compared to the other 30 km/h image.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Speed limit (100km/h)                         |
| .00                   | Speed limit (30km/h)                          |
| .00                   | Speed limit (20km/h)                          |
| .00                   | Speed limit (120km/h)                         |
| .00                   | Speed limit (50km/h)                          |

##### Image 5

My classifier predicted a stop sign (which is also bright red with a whitish mark in the middle), while No entry was second on the list. This suggests my classifier might have trouble with words.

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .99                   | Stop                                          |
| .00                   | No entry                                      |
| .00                   | Yield                                         |
| .00                   | No passing                                    |
| .00                   | Keep right                                    |
