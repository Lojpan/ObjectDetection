import os
import argparse
import numpy as np
import sys
import time
from time import sleep
from DB_Manager import Database
from Video_Manager import VideoStream
import importlib.util

# This class handles all functions and variables used to detect objects.
class object_detector:
        # create parser
    #def __init__(self,model, graph, labels, )
    """MODEL_NAME = ""
    GRAPH_NAME = ""
    LABELMAP_NAME = ""

           # Get path to current working directory
    CWD_PATH = ""
    PATH_TO_LABELS ="""""


        # Parse args
        #args = parser.parse_args()


    ### Define path to working directory and folders containing the model we want to used for object detection and labels used for classification
    def define_path():
        # Print to console for debug purposes
        print("Setting path...")
    
        # Get path to current working directory
        #CWD_PATH = os.getcwd()
        # Path to .tflite file, which contains the model
        #object_detector.PATH_TO_CKPT = os.path.join(object_detector.CWD_PATH,object_detector.MODEL_NAME,object_detector.GRAPH_NAME)
        # Path to label map file
        #PATH_TO_LABELS = os.path.join(object_detector.CWD_PATH,object_detector.MODEL_NAME,object_detector.LABELMAP_NAME)
    
    # Define and parse input arguments
    def parse():
    
        # Print to console for debug purposes
        print("Parsing innput args...")
        # create parser
        #parser = object_detector.parser
        parser = argparse.ArgumentParser()
        
        # add input argguments. You can set your own arguments
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            required=True)
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='1280x720')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')
        return parser

        
    def import_tf_runtime():        
        # Look for tflite_runtime pkg
        pkg = importlib.util.find_spec('tflite_runtime')
        # If pkg is found then import tf lite interpeter from tf runtime.
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            #if object_detector.use_TPU:
            #    from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            #if object_detector.use_TPU:
            #    from tensorflow.lite.python.interpreter import load_delegate
                # If using Edge TPU, assign filename for Edge TPU model
                
        """if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'"""
    
    def get_labels(PATH_TO_LABELS):
        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detector/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])
    
    def load_tf_model(use_TPU,PATH_TO_CKPT, PATH_TO_LABELS, interpeter):
        # If using Edge TPU, use special load_delegate argument
        
        """if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                      experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)
        #return interpreter"""

        interpreter.allocate_tensors()
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5
        
    def initialize_detection():
        object_detector.parse()
        print("importing tf_runtime...")
        object_detector.import_tf_runtime()
        #object_detector.define_path()



            
    def start(input_data):
        """# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std"""

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    
    def draw_inference_box():
    # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index    
                
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0) and (object_name == "person" or object_name == "keyboard")):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                #Capture fram and save locally
                VideoStream.captureObject(object_name, frame)
                


    
        
