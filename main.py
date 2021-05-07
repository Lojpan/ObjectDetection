import sys
import os
import time
from time import sleep
import importlib.util
from DB_Manager import Database
from Object_Detector import object_detector
from Video_Manager import VideoStream


print("importing tf_runtime...")
#object_detector.import_tf_runtime()
pkg = importlib.util.find_spec('tflite_runtime')
# If pkg is found then import tf lite interpeter from tf runtime.
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    print("ERROR: Pkg not found!")

# parse input args
parser = object_detector.parse()
print("Parsing done...")

#initialize tensorflow lite runtime
#object_detector.initialize_detection()
# Parse args
args = parser.parse_args()

# set values
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)



interpreter = tflite_runtime.Interpreter(model_path=PATH_TO_CKPT)

print("Fetching labels...")
object_detector.get_labels(PATH_TO_LABELS)
print("Loading model...")
object_detector.load_tf_model(use_TPU, PATH_TO_CKPT, PATH_TO_LABELS, interpreter)



#initialize stream
stream = VideoStream.init_stream()

#open streaming
stream.start()

#Create resizable Window
cv2.namedWindow('Object Detector DYI', cv2.WINDOW_NORMAL)

# Connect to DB
db_connection = Database.connect()

cursor = Database.createCursor(db_connection)

while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = stream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Start detector
    object_detector.start(input_data)
  
    # Fetch img from local folder
    img = Database.getImg()
    
    ### INSERT DATA TO DB
    database.insertData(object_detector.object_name, img, cursor)

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        db_connection.close()
        break

# Clean up
cv2.destroyAllWindows()
stream.stop()

