import streamlit as st
import os
import cv2
import supervision as sv
from ultralytics import YOLO
from time import sleep
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image
import tempfile
import gdown


st.title('YOLOV8 Object Counting')

#Constants
MODEL_NAME = 'yolov8n.pt' 
HOME = os.getcwd()
MODEL_DEFAULT_PATH = os.path.join(HOME, 'models', MODEL_NAME)
SOURCE_VIDEO_PATH=os.path.join(HOME, 'data/raw', 'bgl.mp4')
TARGET_VIDEO_PATH = os.path.join(HOME, 'data/processed', 'temp.mp4')

def download_from_gdrive(gdrive_url, output_path):
    gdown.download(gdrive_url, output_path, quiet=False)

if not os.path.exists(SOURCE_VIDEO_PATH):
    download_from_gdrive('https://drive.google.com/uc?id=1Zv-i5bj5wi22URGf3otlOSHJzdkSi7V_', SOURCE_VIDEO_PATH)
   
# function to load model with cache enabled for 60 minutes
@st.cache_resource(ttl=60*60) 
def load_model(path):
    return YOLO(path)

# Loading YOLO model
placeholder = st.empty()
if os.path.exists(MODEL_DEFAULT_PATH):
    model = load_model(MODEL_DEFAULT_PATH)
else:
    placeholder.info('Model does not exist, downloading and this may take a while', icon="ℹ️")
    model = load_model(MODEL_NAME)
    os.rename(MODEL_NAME, MODEL_DEFAULT_PATH)

placeholder.success('The model is loaded successfully')
sleep(1)
placeholder.empty()

#Show mode size info
st.sidebar.markdown(f"Model Type: {MODEL_NAME[:-3]}")
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space


# Video File Uploader and use a temp file to store the uploaded file and delete it after the process automatically
uploaded_file = st.sidebar.file_uploader("Upload a video file, it should be less than 20MB for a performance concern", type=["mp4"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(uploaded_file.read())
    SOURCE_VIDEO_PATH = tfile.name
# Adding space between sidebar items
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space

#adjust confidence threshold
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space

# Select desired detection classes
class_names_dict = model.model.names
selected_class_names = st.sidebar.multiselect("elect Desired Classes", list(class_names_dict.values()), ['car','person'])
selected_class_ids = [k for k, v in class_names_dict.items() if v in selected_class_names]


# Show video info
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space
st.sidebar.markdown(f"Source Video Information:")
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.write(f"width\n{video_info.width}")
with col2:
    st.write(f"height\n{video_info.height}")
with col3:
    st.write(f"#frames\n{video_info.total_frames}")

# Point Configuration
# TODO: set proper default value for line start and line end
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space
st.sidebar.markdown(f"Line Counter Configuration:")
LINE_START = tuple(map(int, st.sidebar.text_input("Starting Point (x,y)", "180,50").split(',')))
LINE_END = tuple(map(int, st.sidebar.text_input("Ending Point (x,y)", "180,1230").split(',')))

# Check x from either LINE_START or LINE_END should be smaller than  video_information.width otherwise raise error
if LINE_START[0] > video_info.width or LINE_END[0] > video_info.width:
    st.error('x from either LINE_START or LINE_END should be smaller or equal to video width')
    st.stop()
# Check y from either LINE_START or LINE_END should be smaller than  video_information.height otherwise raise error
if LINE_START[1] > video_info.height or LINE_END[1] > video_info.height:
    st.error('y from either LINE_START or LINE_END should be smaller or equal to  video height')
    st.stop()


# Box annotator
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space
st.sidebar.markdown(f"Box Annotator Configuration:")
box_annotator_thickness = st.sidebar.number_input("Box Thickness", min_value=1, value=1)
box_annotator_text_thickness = st.sidebar.number_input("Box Text Thickness", min_value=1, value=1)
box_annotator_text_scale = st.sidebar.number_input("Box Text Scale", min_value=0.1, max_value=1.0, value=0.5)


# Line counter annotator
st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Add space
st.sidebar.markdown(f"Line Counter Annotator Configuration:")
line_thickness = st.sidebar.number_input("Line Thickness", min_value=1, value=1)
line_text_thickness = st.sidebar.number_input("Line Text Thickness", min_value=1, value=1)
line_text_scale = st.sidebar.number_input("Line Text Scale", min_value=0.1, max_value=1.0, value=0.5)

# Initialize Streamlit placeholder
frame_placeholder = st.empty()

def callback(frame: np.ndarray, index:int) -> np.ndarray:
   
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_class_ids)]
    detections = detections[detections.confidence > confidence_threshold]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    #TODO:Adding progress bar
    box_annotated_frame=box_annotator.annotate(scene=frame.copy(),
                                    detections=detections,
                                    labels=labels)
    # update line counter
    line_zone.trigger(detections)
    # return frame with box and line annotated result

    line_counter_annotated_frame = line_zone_annotator.annotate(box_annotated_frame, line_counter=line_zone)

    _, buffer = cv2.imencode('.jpg', line_counter_annotated_frame)
    frame_as_bytes = buffer.tobytes()
    frame_placeholder.image(frame_as_bytes, channels="BGR", use_column_width=True, output_format="auto")
                            
    return  line_counter_annotated_frame

# Real-time Video Stream

LINE_START = sv.Point(*LINE_START)
LINE_END = sv.Point(*LINE_END)
# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh= 0.25, track_buffer = 30,match_thresh = 0.8,frame_rate =30)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=box_annotator_thickness, text_thickness=box_annotator_text_thickness, text_scale=box_annotator_text_scale)

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=line_thickness, text_thickness=line_text_thickness, text_scale=line_text_scale)

# process the whole video
sv.process_video(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)