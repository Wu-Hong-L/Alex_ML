#!/usr/bin/python
import io
import os
import cv2
import open3d as o3d
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
from helper_class import *
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

CUSTOM_MODEL_NAME = 'my_ssd_resnet50_test_1' 
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-17')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Width, height, frame parameters
WIDTH = 640
HEIGHT = 480
FRAMES = 60

# Configure (depth) and color streams
pipeline = rs.pipeline()

# Create a config object:
config = rs.config()

config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAMES)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FRAMES)

# Start streaming
print("[INFO] Start streaming...")
pipeline.start(config)

try:
    while True:

        # Align RGB and Depth frames
        frames = pipeline.wait_for_frames()
        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        # Obtain Camera Intrinsic Parameters 
        intrinsics = aligned_frames.profile.as_video_stream_profile().intrinsics
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        # Obtain RGB and Depth frame separately 
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        

        if not color_frame or not depth_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Object Detection
        input_tensor = tf.convert_to_tensor(np.expand_dims(color_image, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = color_image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)
        
        # Display Detected Image
        cv2.namedWindow('RealSense',cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense',image_np_with_detections)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break        
        
        height, width, channels = color_image.shape
        ymin, xmin, ymax, xmax = detections['detection_boxes'][0]
        bbox = [int(ymin*height), int(xmin*width), int(ymax*height), int(xmax*width)]
        
        
        if (detections['detection_scores'][0] > 0.85):
            print("[INFO] Stairs Detected")

            # Crop RGB Image
            color_crop_img_np = color_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # Crop Depth Image
            depth_img_np = np.asanyarray(depth_frame.get_data())
            depth_crop_img_np = depth_img_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # Feature Extraction
            img_color = o3d.geometry.Image(color_crop_img_np)
            img_depth = o3d.geometry.Image(depth_crop_img_np)

            #Create RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, 
                                                                        img_depth,convert_rgb_to_intensity=False)


            # Create Point Cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)


            # Downsample the point cloud
            pcd = pcd.voxel_down_sample(voxel_size=0.04)

            counter = 0

            # Initialize parameters 
            min_ratio = 0.1

            # Using RANSAC to detect planes 
            results = detect_planes(pcd, min_ratio)

            # Dimension calculation
            tread, riser = feature_calc(results)


            print("[INFO] Tread dimensions = %.2f m"%tread)
            print("[INFO] Riser dimensions = %.2f m"%riser)
            
finally:
    pipeline.stop()
    print("[INFO] Streaming Stopped.")
    cv2.destroyAllWindows()