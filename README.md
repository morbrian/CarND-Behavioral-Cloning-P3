# Project-3: Behavioral Cloning

The third project in the first 3 month session of the Udacity Self Driving Car series.

* [Introduction](#introduction)
* [Background](#background)
* [Approach](#approach)

---
## Introduction

This project is a collection of data tools and deep learning algorithms designed to 
assist a data scientist with developing a model for effectively predicting steering
angles of a vehicle based on imagery provided by front facing cameras mounted on 
the vehicle. This work was initially created based while completing the third project 
in the first 3 month session  of the Udacity program focuses on Deep Learning  and 
Computer Vision. This project in particular is focused on implementing a Deep Learning 
neural network to teach an  unmanned vehicle to stay on the road by using an approach 
called behavioral cloning.

In [Background](#background) we discuss the project goals and research topics recommended
as we began work on the project. In [Approach](#approach) we discuss the high level
approach we used to develop a successful solution to the problem of automating the
steering of a self-driving car based on image input. Each of the sections after
[Approach](#approach) provides greater detail on our implementation of the high
level core areas of work for the project, [Model Development](#model-development),
[Data Collection](#data-collection) and [Data Preparation](#data-preparation).

---
## Background

The term *behavioral cloning* describes a strategy by which a computer can learn to observe 
characteristics of a particular behavior and then mimic the behavior on its own. In this
project the observed behavior is simple instance of driving a vehicle on an empty road,
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

We started out by implementing the [NVidia Model][nvidia-model] and then testing it on about
40,000 data samples we collected by manually driving the vehicle around the simulated track.
The model promptly guided the car onto the curb where it was stuck. After some time
debugging, we copied the [Commai][commai-model] resulting in a similarly stuck vehicle.

After much experimentation and tips from the Slack forum and [Udacity][udacity] mentor, we came to understand 
the process as being far more sensitive to the sample data than to the choice of deep learning model. 
Our first attempts at data collection had relied exclusively on the keyboard which resulted in choppy 
steering angles, and significantly more straight steering angles than appropriate for the track.

We were able to acquire two alternative datasets, one provided by [Udacity][udacity] and another provided 
by a student on the course Slack channel. Together, these provided nearly 130,000 images which
a wide variety of steering angles more appropriate for the track than what we could manually generate.

Training on this new data helped the vehicle drive straight into the lake.

Up until this point we had done only put a small amount of effort into data preparation, believing at
the time that we would be spending more time tweaking model parameters than manipulating the data. Our
initial data prep steps included flipping the image along the vertical access and associating it with
the negative of the original steering angle to get an equal number of left/right images. We also 
filtered out most of the zero angle samples. At the time we believed we had balanced the dataset
since we had a fairly balanced set of data for angles between -0.25 and 0.25, and still ok
extending to -0.5 and 0.5.

What was missing was examples of steering back toward the center of the road after drifting
to the sideline. Without examples of what not to do, or at least how to escape an undesirable
state, the vehicles was fairly happy to just drive straight off the road into the lake,
it had not been provided with examples to learn that it should avoid drifting to or
crossing the road lines.

Using a PS4 controller we were able to create smoother steering angles, and we used this
to generate more data in the simulator with a focus on recovery driving. We made a few
laps recording instances of the car driving away from the sideline and added this
new recovery data to the existing data we had obtained.

In addition to the recovery data we updated our sample selection algorithm to pick training samples
in a more balanced way, making roughly as likely to return a sample with a steering angle
anywhere between -1.0 and 1.0, rather than heavily favoring the smaller valued angles.

With that, we finally had a working model that could drive the vehicle around track1 without incident!

---
## Model

TODO 

---
## Data Collection

TODO

---
## Data Preparation

TODO


[//]: # (Research References)
[nvidia-model]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
[commai-model]: https://github.com/commaai/research/blob/master/train_steering_model.py
[udacity]: https://classroom.udacity.com/nanodegrees/nd013