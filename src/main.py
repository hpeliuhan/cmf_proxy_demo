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
                if predictions[0][0] >= smoke_threshold:
                    sample_path = "image.jpg"
                    sample.save(sample_path)
                    plugin.upload_file(sample_path, timestamp=sample.timestamp)
                    plugin.publish("classification.certainty", float(predictions[0][0]),
                                    timestamp=sample.timestamp,
                                    meta={"camera": f'{camera_src}'})
                    
                    logging.info(f"Smoke detected in frame at {sample.timestamp},with probability {predictions[0][0]}")
                #time.sleep(1 / 30)


if __name__ == "__main__":
    FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
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

    PYWAGGLE_dir = os.environ.get("PYWAGGLE_LOG_DIR", "test")
    os.environ["PYWAGGLE_LOG_DIR"] = PYWAGGLE_dir

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
        raise

    # Process the video
    process_video(video_path, input_size, 0.9)


