# Project-3: Behavioral Cloning

The third project in the first 3 month session of the Udacity Self Driving Car series.

* [Introduction](#introduction)
* [Background](#background)
* [Approach](#approach)
    * [Early Failed Attempts](#early-failed-attempts)
    * [Problem Identifeid](#problem-identified)
    * [Finally Success](#finally-success)
* [Model](#model)
* [Data Collection](#data-collection)
* [Data Preparation](#data-preparation)
* [Training Process](#training-process)
* [Augmentation Through Jittering](#augmentation-through-jittering)
* [How To Run Code](#how-to-run-code)

---
## Introduction

This project is a collection of data tools and deep learning algorithms designed to 
assist a data scientist with developing a model for effectively predicting steering
angles of a vehicle based on imagery provided by front facing cameras mounted on 
the vehicle. This work was initially created while completing the third project 
in the first 3 month session of the Udacity program focused on Deep Learning and 
Computer Vision. This project in particular is focused on implementing a Deep Learning 
neural network to teach an  unmanned vehicle to stay on the road using an approach 
called behavioral cloning.

In [Background](#background) we discuss the project goals and research topics recommended
as we began work on the project. In [Approach](#approach) we discuss the high level
approach we used to develop a successful solution to the problem of automating the
steering of a self-driving car based on image input. Each of the sections after
[Approach](#approach) provides greater detail on our implementation of the high
level core areas of work for the project, [Model Development](#model-development),
[Data Collection](#data-collection) and [Data Preparation](#data-preparation).

In the final sections we discuss the capabilities provided by the program files
and how to run them, [How To Run Code](#how-to-run-code).

---
## Background

The term *behavioral cloning* describes a strategy by which a computer can learn to observe 
characteristics of a particular behavior and then mimic the behavior on its own. In this
project the observed behavior is simple instances of driving a vehicle on an empty road,
and the method of observation is via three front facing cameras mounted on the vehicle
to capture images of the road and landscape ahead along with the steering angle of 
the vehicle at the moment the images was captured.

Udacity students are provided with a computer simulator in which they can manually 
drive a vehicle around a track in a 3D graphics modeled landscape. Driving the virtual 
car around the track like one would do in a video game provides students with the means 
to collect data for a variety of situations the car might encounter in the virtual world.

The expectation is that students will be able to use the recorded data in order to
train an algorithm to produce steering angles that will keep the car from veering
of the road.

---
## Approach

It felt easy to get started on this project. Keras makes it simple to layer together an
arbitrary network model and the simulator seemed to make data collection simple. But our early
optimism slowly faded after a series of failed experiments, only to return again once
we learned to secrets to the problem.

### Early Failed Attempts

We started out by implementing the [NVidia Model][nvidia-model] and then testing it on about
40,000 data samples we collected by manually driving the vehicle around the simulated track.
The model promptly guided the car onto the curb where it was stuck. After some time
debugging, we copied the [Commai][commai-model] resulting in a similarly stuck vehicle.

After much experimentation and tips from the Slack forum and [Udacity][udacity] mentor, we came to understand 
the process as being far more sensitive to the sample data than to the choice of deep learning model. 
Our first attempts at data collection had relied exclusively on the keyboard which resulted in choppy 
steering angles, and significantly more straight steering angles than appropriate for the track.

We were able to acquire two alternative datasets, one provided by [Udacity][udacity] and another provided 
by a student on the course Slack channel. Together, these provided nearly 130,000 images with
a wide variety of steering angles more appropriate for the track than what we could manually generate.

Training on this new data helped the vehicle drive straight into the lake.

### Problem Identified

Up until this point we had only put a small amount of effort into data preparation, believing at
the time that we should spend more time refining the model than manipulating the data. Our
initial data prep steps included flipping the image along the vertical access and associating it with
the negative of the original steering angle to get an equal number of left/right steering images. We also 
filtered out most of the zero angle samples. At the time we believed we had balanced the dataset
since we had a well balanced set of data for angles between -0.25 and 0.25, and still pretty good
extending to -0.5 and 0.5. While driving we had not noticed a need for much larger angles than that.

We discovered we were missing examples of steering back toward the center of the road after drifting
to the sideline. Without examples of what not to do, or at least how to escape an undesirable
state, the vehicles was fairly happy to just drive straight off the road into the lake,
it had not been provided with examples to learn about avoiding drifts to the side or
completely crossing the road lines.

We also realized that although large angles appeared to be rarely needed on the track,
having few to no examples of it in the training data meant the vehicle could not learn
what to do in the less common, but still occasionally necessary instances when a larger
steering angle was actually needed.

### Finally Success

Using a PS4 controller we were able to create smoother steering angles, and we used this
to generate more data in the simulator now with a focus on recovery driving. We made a few
laps recording instances of the car driving away from the sideline and added this
new recovery data to the existing data we had obtained.

In addition to the recovery data we updated our sample selection algorithm to pick training samples
in a more balanced way, making it roughly as likely to return a sample with a steering angle
anywhere between -1.0 and 1.0, rather than heavily favoring the mid-range valued angles.

With that, we finally had a working model that could drive the vehicle around track1 without incident!

---
## Model

We started with the [NVidia Model][nvidia-model], and once we learned how to train it successfully we
added some layers of our own to experiment. We use a lambda layer to perform initial normalization,
and then apply cropping to focus more on the road than the horizon.

The [NVidia paper][nvidia-model] did not state what they used for activation. We decided to specify `elu`
activation as part of each convolution layer, inspired by seeing the [Commaai model][commaai-model] use of
`elu` in their approach. `Elu` is described as being more useful than `Relu` for regression problems like the
steering prediction problem, where we are not predicting specific classes but rather choosing roughly
good enough steering angles from continuous range to keep the vehicle on the road. 

It was also not clear to us whether the [NVidia paper's][nvidia-model] description of the model applied 
only to the usable result or also to the training process. For example we did not see mention of Dropout
in the paper, but we know those are turned off outside of training. We added Dropout to our model in
an effort to reduce overfitting to track-1.

Our final model is essentially a deeper version of the [NVidia][nvidia-model]. We added additoinal
convolution2d layers in the hopes of giving the architecture a better chance at pulling out more
important features.

In experimentation we added a `LocallyConnected2D` layer with the intuition that allowing the model 
to evaluate the parameters in isolation before entering the flatten layer. This did not perform
much differently, but more than doubled the number of parameters so we removed it.

We did keep a `MaxPooling2D` layer in the model, as it seemed to perform a bit better in the
couple trials we observed.

When discussing whether the model performed well or not, we are using the observation of the vehicle
staying in the middle of the road as our assessment. It is also important to note that there is 
a significant amount of randomness built into the data generator, so one or two runs of a particular
model are not enough to make a fair comparison of performance with other network models.

Number Epochs: 40
Batch Size: 256
Samples per Epoch: 20,000
Training Duration: 4215.032s

        ____________________________________________________________________________________________________
        Layer (type)                     Output Shape          Param #     Connected to                     
        ====================================================================================================
        lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
        ____________________________________________________________________________________________________
        cropping2d_1 (Cropping2D)        (None, 95, 320, 3)    0           lambda_1[0][0]                   
        ____________________________________________________________________________________________________
        convolution2d_1 (Convolution2D)  (None, 88, 313, 12)   2316        cropping2d_1[0][0]               
        ____________________________________________________________________________________________________
        convolution2d_2 (Convolution2D)  (None, 42, 155, 24)   7224        convolution2d_1[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_3 (Convolution2D)  (None, 19, 76, 36)    21636       convolution2d_2[0][0]            
        ____________________________________________________________________________________________________
        dropout_1 (Dropout)              (None, 19, 76, 36)    0           convolution2d_3[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_4 (Convolution2D)  (None, 9, 37, 48)     15600       dropout_1[0][0]                  
        ____________________________________________________________________________________________________
        dropout_2 (Dropout)              (None, 9, 37, 48)     0           convolution2d_4[0][0]            
        ____________________________________________________________________________________________________
        convolution2d_5 (Convolution2D)  (None, 4, 18, 64)     27712       dropout_2[0][0]                  
        ____________________________________________________________________________________________________
        maxpooling2d_1 (MaxPooling2D)    (None, 2, 9, 64)      0           convolution2d_5[0][0]            
        ____________________________________________________________________________________________________
        dropout_3 (Dropout)              (None, 2, 9, 64)      0           maxpooling2d_1[0][0]             
        ____________________________________________________________________________________________________
        flatten_1 (Flatten)              (None, 1152)          0           dropout_3[0][0]                  
        ____________________________________________________________________________________________________
        dense_1 (Dense)                  (None, 1164)          1342092     flatten_1[0][0]                  
        ____________________________________________________________________________________________________
        dropout_4 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
        ____________________________________________________________________________________________________
        dense_2 (Dense)                  (None, 100)           116500      dropout_4[0][0]                  
        ____________________________________________________________________________________________________
        dropout_5 (Dropout)              (None, 100)           0           dense_2[0][0]                    
        ____________________________________________________________________________________________________
        dense_3 (Dense)                  (None, 50)            5050        dropout_5[0][0]                  
        ____________________________________________________________________________________________________
        dropout_6 (Dropout)              (None, 50)            0           dense_3[0][0]                    
        ____________________________________________________________________________________________________
        dense_4 (Dense)                  (None, 10)            510         dropout_6[0][0]                  
        ____________________________________________________________________________________________________
        dropout_7 (Dropout)              (None, 10)            0           dense_4[0][0]                    
        ____________________________________________________________________________________________________
        dense_5 (Dense)                  (None, 1)             11          dropout_7[0][0]                  
        ====================================================================================================
        Total params: 1,538,651
        Trainable params: 1,538,651
        Non-trainable params: 0

---
## Data Collection

The data collection process should involve driving around the track with a PS4 controller. We spent
a significant amount of time driving around the track with a keyboard, and later with a mouse. The
keyboard and mouse options were both error prone and generated examples of bad driving which lead
the vehicle off the road. 

Our final data set is a combination of data provided by others with a number of specific examples
we added. In the data metrics below, number of samples counts each camera image as a separate image.

1. [Udacity][udacity]

This is the original data provided to students by Udacity.

        ===== track1-given-pair_log =====
        data_count: 24108
        data range: (-0.94269539999999996, 1.0)
        hist: [   12     3     6    18    33   186   234   900  1419  2514 15255  2343   483   516   126    36    15     3     0     6]
        bin_edges: [-1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]

![histogram][udacity-data]

2. [bc-track-1][slack-carnd]: 104,109 samples

This is a large dataset shared by a student on the Slack channel. At the time of this write up we are not able to 
find the student's name because Slack limits searches to recent posts, but the zip file was called bc_track_1.zip

        ===== bc-track-1-pair_log =====
        data_count: 104109
        data range: (-0.89387939999999999, 0.96590160000000003)
        hist: [    0    15    21    54   153   426  1599  6339 10887 10920 70326  1302   981   483   273   165    99    51     9     6]
        bin_edges: [-1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
        
![histogram][slack-student]

3. **self recovery data**: 26,784 samples

This is data we generated ourselves with a PS4 controller to create examples of the vehicle steering away from
the side of the road.

        ===== recover-pair_log =====
        data_count: 26784
        data range: (-0.95214849999999995, 1.0)
        hist: [  732   186   147   135   117    90    96    96   123    81 23748    60    99    60    84    54    60    78   105   633]
        bin_edges: [-1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]

![histogram][self-recovery]

4. **self first curve**: 4,836 samples

This dataset is focused almost exclusively on the first curve in the track and was created when we were
initially just trying to get the vehicle that far.

        ===== first-curve-pair_log =====
        data_count: 4836
        data range: (-0.39707150000000002, 0.34921999999999997)
        hist: [   0    0    0    0    0    0  126  210  642 1878 1950   18    6    6    0    0    0    0    0    0]
        bin_edges: [-1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]

![histogram][self-first-curve]

5. **Total Sample Summary**: 159,837 samples

This final dataset is the combination of the previous 4 datasets.

        ===== moriarty-borrowed-and-augmented-pair_log =====
        data_count: 159837
        data range: (-0.95214849999999995, 1.0)
        hist: [   744    204    174    207    303    702   2055   7545  13071  15393 111279   3723   1569   1065    483    255    174    132    114    645]
        bin_edges: [-1.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]

![histogram][combined]

---
## Data Preparation

In our final iterations of work on this project we used a data generator to perform data generation in realtime
during training. We will talk more about the realtime data generation and augmentation in the next section on
[Training Process](#training-process), while this section will discuss some of the tools and approaches we
used in our initial experimentation with the data.

From the histograms in the section on [Data Collection](#data-collection) it is obvious that the collected data
is heavily biased toward samples with zero steering angles, as well as negative steering angles due to driving
around the track mainly in a counter-clockwise direction.

In our initial approach to working with the data we wrote a number of algorithms to filter and augment the data,
and then write it to disk in order to train on pre-processed data with no further augmentation during training.

Although that approach began to feel too manually intensive later once we knew what to look for,
it did help prove out the basic building blocks of what would later become our realtime data generator. It was
useful to collect the pre-processed data on disk because it help us review and gather metrics on exactly 
what we planned to train with, plus it meant there would be minimal randomization during training outside
of the initial shuffling of the data. We believe minimal randomization during training is likely to help
reduce confounding results when comparing various training networks, but rigorous comparisons were not a 
concern for the goals this project.

We filtered out 95% of the zero valued angles. We then wanted to balance the left/right steering angles
in the data.  One way to do this would be to drive around the track in the reverse direction, however it was 
less work to use the existing images and write code to flip them on the vertical axis and multiply the angle
by -1.0 to. These first two steps at least put our data into a roughly standard bell curve histogram.

Next we wanted to ensure we had a roughly equivalent number of samples of all possible steering angles
between -1.0 and 1.0. To accomplish this, we divided up the -1.0 to 1.0 range into 20 equal range bins.
We then chose 200 examples with a steering angle in each bin range. For this, we decided it was ok if
some of the random selections were identical to fill up a given bin. It was more important to have an equal
number of samples in each bin than for the samples to be unique.

We also noticed the images where almost half filled up with horizon features like trees and sky,
so we cropped off the top of the image. Later we would perform this crop as a layer in our network,
but at this stage in development we were still reviewing our image results on disk.

We ran this process once for each of three of the datasets above (we left out the first-curve data),
and then combined the 2000 balanced samples from each result to produce a combined 6000 sample dataset.

Now our dataset was truly balanced, and relatively small at 6000 samples compared to the original almost
160,000 sized sample set. We knew some of our samples were duplicates, so to overcome this during training
we generated additional data using the Keras `ImageDataGenerator`.

Our first successful automated drive around the track used the [NVidia Model][nvidia-model] with the 2000
data samples described above and an additional 8000 realtime generated images. The realtime generation
involved shifting the images by up to 5% on the vertical or horizontal axes, shifting the color channels,
and shearing the images, all provided by the Keras ImageDataGenerator.

This exploration informed us of the key components needed to train the vehicle, and is what lead to further
development on our own realtime data generator, described in the next section.

---
## Training Process

In our final approach we use the large 160,000 sized sample set as a data pool to select from,
and we do all the work of data preparation and augmentation during training.

The following pseudo-code describes how the realtime data generator works:

        big-datset = read angles/image-names from file (each pair is a sample)
        filtered-data = filter 98% of zero valued angle samples from big-dataset
        bin_edges = divide the range -1.0 - 1.0 into 20 bins with 21 edges
        while loop_forever:
            for i in range(batch_size):
                random_edge = random.randint(0, len(bin_edges)-2)
                pick_i = keep sampling filtered-data until we find a sample in range of bin
                new_angle, new_image = jitter_data(random shifts, random flips, random noise)
                extend images, angles with new_image, new_angle

            yield images, angles
          
The first two layers of our deep neural network then handle normalizing over the color channels,
and cropping the image.

### Augmentation through Jittering

It turned out to be impractical to use all of the jittering methods we implemented. They are all
useful for experimentation and we are planning to explore and enhance the `shade` feature as a
potential way to produce a model that might work on Track-2 without including any Track-2 data
in the training set.

---
We will be working with variations of this original unmodified image sample:

![jiter][jitter-original] 

---
A flipped image is simply flipped on the vertical axis.

![jitter][jitter-flip]

---
A blurred image has a gaussian blur applied.

![jitter][jitter-blur]

---
A shifted image is repositioned in the vertical and or horizontal direction,
and randomly picks between `nearest` or `wrap` modes for to fill in the empty space.
The image below used `wrap`

![jitter][jitter-shift]

---
A shifted channel image maintains the same position, but recolors based on the new
color arrangement.

![jitter][jittter-shift-channel]

---
An image with noise applied is summed with a randomly generated array of integers.

![jitter][jitter-noise]

---
A sheared image gets smeared to the left or right, so the terrain leans toward the shear.

![jitter][jitter-shear]

---
When shade is applied, it is meant to simulate  unpredictable shade or lighting differences
anywhere in the image.

![jitter][jitter-shade]

## How To Run Code

Two of the four code files in the project were provided by Udacity as a means of interacting
with the simulator. We did not modify these (`drive.py`, `video.py`)for our final submission.

### Data Analysis

The `prep.py` program provides the core data generation and augmentation algorithms used by 
 the model during trainign. By itself, `prep.py` supports two useful capabilities.

First, assuming a directory with `driving_log.csv` and `IMG` paths is available,
 the `prep.py` file will squash the multi-column `driving_log.csv` output into 
 a dual column csv file with a pair row for every image and steering angle
 saved to `pairs_log.csv` next to `driving_log.csv`. It will also generate a
 and save a histogram png file, and create a `metrics.txt` file describing
 the basic distribution of the data.
 
 THe following is an example:
 
        python prep.py -d /path/to/driving-folder

### Jittering Demo

The `prep.py` program also supports demoing an example of each of the augmentation
approaches it has implemented. The jittering demo will accept as input any image
with the same input dimensions as what the simulator produces, and it will output
several jittered images into a specified directory.

The following is an example:

        python prep.py -j True --jitter_input sample-image.jpg --jitter_output ./output-path

## Model Training

The `model.py` program is where our model code is. We have three models available to try out
in the code. The [Commaai Model][commai-model], the [NVidia Model][nvidia-model] and our own
custom model, which is the default.

Running a model requires a directory to be specified where the simulator saved its data.

        python model.py -d /path/to/simulator-output-folder
        
The `moriarty` model is the default model. Trying an alternative model is possible
by specifying the `-m` option and the model name. For example, the command below
runs the NVidia model.

        python model.py -d /pathto/simulator-output-folder -m nvidia

---
[//]: # (Research References)

[nvidia-model]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
[commai-model]: https://github.com/commaai/research/blob/master/train_steering_model.py
[udacity]: https://classroom.udacity.com/nanodegrees/nd013
[slack-carnd]: https://carnd.slack.com/messages/p-behavioral-cloning/

[//]: # (Histo References)

[udacity-data]: ./doc/histo/track1-given-pair_log.png "Histogram of Udacity provided data"
[slack-student]: ./doc/histo/bc-track-1-pair_log.png "Histogram of Slack student's data"
[self-recovery]: ./doc/histo/recover-pair_log.png "Histogram of self recovery data"
[self-first-curve]: ./doc/histo/first-curve-pair_log.png "Histogram of first-curve data"
[combined]: ./doc/histo/moriarty-borrowed-and-augmented-pair_log.png "Histogram of combined data"

[//]: # (Jitter References)

[jitter-original]: ./doc/jitter-demo/sample.png "Original image before jittering"
[jitter-blur]: ./doc/jitter-demo/blur.png "Jitter blurred image"
[jitter-flip]: ./doc/jitter-demo/flip.png "Jitter flipped image"
[jitter-noise]: ./doc/jitter-demo/noise.png "Jitter gaussian noise image"
[jitter-shade]: ./doc/jitter-demo/shade.png "Jitter pseudo-shade image"
[jitter-shear]: ./doc/jitter-demo/shear.png "Jitter sheared image"
[jitter-shift-channel]: ./doc/jitter-demo/shift_channel.png "Jitter channel shifted image"
[jitter-shift]: ./doc/jitter-demo/shift.png "Jitter position shifted image"
