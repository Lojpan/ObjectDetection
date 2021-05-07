# ObjectDetection


# Summary
An object recognition project deployed on a raspberry Pi4 with PiCamera. Using tensorflow pre-trained ssd models. 

# Purpose and goal
Goal is to develope an mobile (embedded) unit for outdoor use to detect, monitor wild life and document wild life.

# Resources
Raspberry Pi4
PiCamera
Amazon AWS-RDS (Relational Database Service)


# Funtionality
The Raspberry Pi will be the unit gathering information. This is done using pre-trained models for object detection from googles tensorflow project and Picamera to capture video/pictures. 
When a desired object is detected by the Raspberry Pi a short video or a few pictures are captured and stored in a database (aws-rds). Each time something is detected a box will appear around the object. Data of the object is displayed.
A list of desired objects will be matched against data from detected objects. If they match, object data goes into the database together with a timestamp.
A website/app will then display data from the database. 


# Requirements
Raspberry Pi4 / OS Debian + other dependencies
Picamera/webcam
python 3.7 
mySql connector
AWS-RDS (Amazon web services account needed)
Python Editor (For easier programming, not necessary. AWS-RDS includes a cloud editor) 
tensorflow lite
open cv


