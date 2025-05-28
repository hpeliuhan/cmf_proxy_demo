from waggle.plugin import Plugin
from waggle.data.vision import Camera
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import threading
import os
import time
import shutil
import logging
from cmfsage import cmfsage

def process_video(video_path, input_size, smoke_threshold=0.5):
    camera_src = "file://" + video_path
    with Plugin() as plugin:
        with Camera(camera_src) as camera:
            for sample in camera.stream():
                frame_resized = cv2.resize(sample.data, input_size)
                frame_normalized = frame_resized.astype("float32") / 255.0
                frame_expanded = np.expand_dims(frame_normalized, axis=0)

                # Perform inference
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], frame_expanded)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                #logging.info(f"Predictions: {predictions}")
                if predictions[0][0] >= smoke_threshold:
                    
                    try:
                        #logging.info(f"saving image to image.jpg")
                        cv2.imwrite("image.jpg", sample.data)

                        #sample.save("image.jpg")
                    except Exception as e:
                        logging.error(f"Error saving image: {e}")
                        continue
                    try:
                        #logging.info(f"uploading image.jpg")
                        plugin.upload_file("image.jpg", timestamp=sample.timestamp)
                    except Exception as e:
                        logging.error(f"Error uploading image: {e}")
                        continue
                    plugin.publish("classification.certainty", float(predictions[0][0]),
                                    timestamp=sample.timestamp,
                                    meta={"camera": f'{camera_src}'})
                    
                    logging.info(f"Smoke detected in frame at {sample.timestamp},with probability {predictions[0][0]}")
                time.sleep(1)


if __name__ == "__main__":
    FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.CRITICAL,
        format=FORMAT,
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    # Path to the folder containing videos downloaded in the Dockerfile
    docker_videos_folder = "/src/"
    
    # Ensure the folder exists
    if not os.path.exists(docker_videos_folder):
        raise FileNotFoundError(f"Videos folder not found at {docker_videos_folder}")

    # Assume the Dockerfile downloads a specific video file (e.g., "video.mp4")
    video_file = "video.mp4"
    video_path = os.path.join(docker_videos_folder, video_file)

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")


    print(f"Processing video: {video_file} ")

    # Initialize the TFLite interpreter
    model_path = "/src/model.tflite"
    print(f"Model path: {model_path}")
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        input_size = (input_details[0]['shape'][2], input_details[0]['shape'][1])
        result_dir = "test/uploads"
        logging.info(f"model loaded, Input size: {input_size}")
    except Exception as e:
        print(f"Error initializing TFLite interpreter: {e}")
        raise FileNotFoundError(f"Model file not found at {model_path}")


    '''cmf sage part'''   
    '''Pipeline name: What's the pipeline to trace'''
    '''Pipeline file: The mlmd file name, mlmd file be default'''
    '''Model path: The location to the model'''
    '''Result path: The location to the inference result'''
    '''Git remote url: The git remote url to the AI repository'''
    params = {
        "pipeline_name": "wildfire-classification",
        "pipeline_file": "mlmd", #this will be the branch name
        "model_path": "model.tflite",
        "result_path": "image.jpg",
        #this can be a directory or a file
        #if this is a file, cmf_sage will use timestamp to track the files changed. Apply higher monitoring frequency
        #if this is a directory, cmf_sage will keep track of the files changed in the directory. Apply lower monitoring frequency
        "git_remote_url": "https://github.com/hpeliuhan/cmf_proxy_demo.git",
        "archiving": True,
        "logging_interval": 6 # seconds, #how often it logs the artifacts
    }

    cmf_logger = cmfsage(**params)
    cmf_logger.start()

    # Process the video
    process_video(video_path, input_size, 0.95)


