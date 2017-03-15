# **Traffic Sign Recognition** 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 

# **1. Files Submitted**

### *1) Submission Files: The project submission includes all required files.*

**Comment**: All required files are included in this repository.
* The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code cells executed and displaying output.
* An HTML or PDF export of the project notebook with the name report.html or report.pdf.
* Any additional datasets or images used for the project that are not from the German Traffic Sign Dataset.
* A writeup report as a markdown or pdf file (this file)

# **2. Dataset Exploration**

### *1) Dataset Summary: The submission includes a basic summary of the data set.*

**Comment**: I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### *2) Exploratory Visualization: The submission includes an exploratory visualization on the dataset.*

**Comment**: Below are 10 random images extracted from the training set (ref. 1):

![Figure 1](10random_classes.png)

Below is a histogram showing the frequency of each class:

![Figure 2](histogram1.png)

The historgram clearly shows significant variability in class representation - some classes < 250 and other classes > 1,500. I addressed the over/under-representation issue by data augmentation (see *Preprocessing* below).

# **3. Design and Test a Model Architecture**

### *1) Preprocessing: The submission describes the preprocessing techniques used and why these techniques were chosen.*

**Comment**: First, I converted the images to grayscale and normalized the signal intensity to (-1, 1) to improve the model preformance.

* The shape of the image after grayscaling/normalization is (32, 32, 1)
* The mean signal intensity of the image is:
  + Training set -0.354081335648
  + Validation set -0.347215411128
  + Test set -0.358215153428

Below are 10 random image data extracted from the training set after grayscaling and normalization
![Figure 3](10random_classes_gn.png)

Second, I augmented the training set to address the over/under-representation issue. Specifically, I used a combination of random translation, random scaling, random warping, and random brightness to increase the minium number of each label to be 1,000.

Random translation:

![Figure 4](transplation.png)

Random scaling:

![Figure 5](scaling.png)

Random warping:

![Figure 5](warping.png)

Random brightness:

![Figure 6](brightness.png)

Below is a comparison between the orignal (top row) and the augmented images (bottom row) of 5 randomly selected images.

![Figure 7](augmented.png)

In the original training set, the minimum number of label was 180. In the augmented training set, I increased it to 1,000. Now the historgram shows more homogeneous distribution of labels. 

![Figure 8](histogram2.png)

In this project I did not use rotation or flips, because it is unlikely that the self-driving would encounter rotated (hopefully) or flipped traffic signs. I also did not use color perturbation because the images are grayscale.


### *2) Model Architecture: The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.*

**Comment**: Initially I used the original LeNet model and simply changed the output to 43 labels instead of 10 numbers. However, after several iterations the test accuracy reached only at 0.834. So I used the modified LeNet model (ref. 2). My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| **Layer 1: Convolution 5x5**     	| 1x1 stride without padding. Output 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride. Output 14x14x6 				|
| **Layer 2: Convolution 5x5**	    |  1x1 stride without padding. Output 10x10x16  	|
| RELU		|         									|
| Max pooling		| 2x2 stride. Output 5x5x16        									|
|	Flatten Layer 2		|	Output 400									|
|	**Layer 3: Convolution 5x5**	|	1x1 stride without padding. Output 1x1x400		|
| RELU		|         									|
|	Flatten Layer 3		|	Output 400							|
|	Concatenate Layer 2 and Layer 3		|	Output 800						|
|	Dropout		|	Output 800						|
|	**Layer 4: Fully connected**		|	Output 120						|
| RELU		|         									|
|	**Layer 5: Fully connected**		|	Output 84						|
| RELU		|         									|
|	**Layer 6: Fully connected**		|	Output 43						|


### *3) Model Training: The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.*

**Comment**: To train the model, I used the following parameters:

* Learning rate at 0.001
* Adam optimizer to minimize the loss of cross-entropy
* Batch size was initially set at 100-128 and had an accuracy >0.90. Since the size of the training set increased to >50,000 by augmentation, the batch size was increased to 500 to take full advantage of the training set. As a result, the model performance improved slightly.
* Number of epochs was set at 100, which was a realistic maximum on my MacBook Air (CPU only)


### *4) Solution Approach: The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.*

**Comment**: My final model results were:

* Validation set accuracy of 0.929 at EPOCH 100
* Test set accuracy of 0.931 > 0.93!

Summary of iterations:
* Initially I used the original LeNet model, but the test accuracy reached only at 0.834. So I used the modified LeNet model (ref. 2).
* The modified LeNet model produced higher test accuracy values than the original LeNet model, but the test accuracy was still < 0.92.
* I removed Layers 4, 5 and 6 (fully connected layers) but the test accuracy did not significantly improve. I put them back in the final model.
* Other parameters that I iterated include learning rate, batch size, droput probability.
 
 
# **4. Test a Model on New Images**

### *1) Acquiring New Images: The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify.*

**Comment**: I downloaded the test set from [**Dataset discussion: German Traffic Signs**](http://forums.fast.ai/t/dataset-discussion-german-traffic-signs/766). I picked 5 random images shown below:

![Figure 9](myimgs.png)

The size of the new images is not homogeneous. In addition, none of the images that I checked was square. The extension of the images is .ppm. I used [ImageJ](https://imagej.nih.gov/) to adjust the size of each of new images to 32x32 to be able to feed them into my model. When I adjusted the size, I did not maintain the aspect ratio to make it more challenging for the model to correctly classify the label. I also changed the image type to .png. 

Then the same preprocessing method (grayscaling and normalization) was applied to the 5 images. There are two potential issues associates with the new 5 images that may make the model perfomance low:

* Red and blue are inverted. For example, the edge of the 1st image ("Vehicles over 3.5 metric tons prohibited") should actually be red, but in this image it is blue. Therefore, the pixel values may be off from those of the training set. However, grayscaling should address at least some of the issue.
* The signal intensity seems higher than those of the traning set. Therefore, the pixel values may be off from those of the training set. However, normalization should address at least some of the issue.


### *2) Performance on New Images: The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.*

**Comment**: Below are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      		| Vehicles over 3.5 metric tons prohibited   									| 
| Speed limit (30km/h)     			| Speed limit (30km/h)										|
| Keep right					| Keep right											|
| Turn right ahead	      		| Turn right ahead					 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This was better than the accuracy on the test set of 93.1%. It is reassuring that my model performs well in real-world examples, but 5 examples are by no means sufficient to determine the overall performance of the model.


### *3) Model Certainty - Softmax Probabilities: The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.*

**Comment**: Below is a figure showing the top five softmax probabilities of the predictions on the 5 new images:

![Figure 10](certainty_pict.png)

Below are bar charts:

![Figure 11](certainty_bar.png)

Each image was classified correctly with 100% certainty. Although I spent a significant amount of time in model iterations, the results looks too good to be true. I suspect there is overfitting here. I should be able to verify it by testing the model with more new sample images.


REFERENCE
1. [Jeremy Shannon's blog](https://medium.com/@jeremyeshannon/udacity-self-driving-car-nanodegree-project-2-traffic-sign-classifier-f52d33d4be9f#.j74ms0lgu)
2. Sermanet P, LeCun Y. Traffic sign recognition with multi-scale convolutional networks, 2011. [pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

