import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
import asyncio
from cmflib import cmf
import tarfile
import subprocess
import hashlib
import json
import asyncssh

def get_model(model_path: str, model_name: str = 'yolov8n.pt'):
    full_model_path = os.path.join(model_path, model_name)
    
    if not os.path.exists(full_model_path):
        print(f"Model not found at {full_model_path}. Downloading...")
        os.makedirs(model_path, exist_ok=True)
        
        # Download the model
        model = YOLO(model_name)
        # Save the model to the specified path
        model.save(full_model_path)
        #print(f"Model saved to {full_model_path}")

        #delete the model in the parent path
        os.remove(model_name)

    else:
        print(f"Model found at {full_model_path}. Loading...")

    # Load the model from local path
    return YOLO(full_model_path)

# Async function to save image and bounding boxes
async def save_frame(frame, boxes, frame_count, output_dir="output", result_queue=None,model=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(output_dir, f"frame_{frame_count}_{timestamp}.jpg")
    bbox_path = os.path.join(output_dir, f"frame_{frame_count}_{timestamp}_bboxes.txt")

    # Save image
    cv2.imwrite(img_path, frame)

    # Save bounding boxes
    with open(bbox_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            label = model.names[int(box.cls)]
            f.write(f"{label},{int(x1)},{int(y1)},{int(x2)},{int(y2)}\n")

    print(f"[Saved] {img_path} and {bbox_path}")

    # Put file paths into the queue
    if result_queue:
        await result_queue.put(img_path)
        await result_queue.put(bbox_path)
        print(f"Added {img_path} and {bbox_path} to queue")

#run infernecing on the video
async def process_video(video_path, output_dir="output", result_queue=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)
        boxes = results[0].boxes

        # Draw boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = box.conf[0]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame_count += 1
        # Save frame and bounding boxes asynchronously
        await save_frame(frame, boxes, frame_count, output_dir=output_dir, result_queue=result_queue,model=model)
        await asyncio.sleep(0)
        
        #await asyncio.sleep(1)

    cap.release()
#arcvhiing the results and do cmf logging yet we are not pushing
async def archive_and_log_results(output_dir="archive", result_queue=None, cmf_queue=None,upload_queue=None, metawriter=None, model_path=None, stop_event=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Log the model
    _ = metawriter.log_model(
        model_path,
        event="input",
        model_framework="yolov8",
        model_type="object_detection",
        model_name="yolov8_object_detection"
    )
    #add model and mode.dvc to the upload queue
    model_dvc_path = f"{model_path}.dvc"
    await upload_queue.put(model_path)
    await upload_queue.put(model_dvc_path)



        
    
    while not (stop_event and stop_event.is_set()):
        await asyncio.sleep(10)
        print("Checking for files to archive...")
        
        if result_queue.empty():
            print("No new files to archive.")
            continue
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tar_path = os.path.join(output_dir, f"archive_{timestamp}.tar.gz")
        
        with tarfile.open(tar_path, "w:gz") as tar:
            while not result_queue.empty():
                file_path = await result_queue.get()
                print(f"Archiving {file_path}")
                if os.path.isfile(file_path):
                    tar.add(file_path, arcname=os.path.basename(file_path))
                    os.remove(file_path)
                result_queue.task_done()
        
        print(f"Archived files to {tar_path}")
        if cmf_queue:
            await cmf_queue.put(tar_path)
        
        # Log the archived file
        while not cmf_queue.empty():
            file_name = await cmf_queue.get()
            _ = metawriter.log_dataset(file_name, event="output")
            cmf_queue.task_done()
        #add the tar file to the upload queue
        tar_dvc_path=f"{tar_path}.dvc"
        await upload_queue.put(tar_path)    
        await upload_queue.put(tar_dvc_path)


def get_file_hash(file_path):
    """Calculate the hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def track_file_changes(cache_dir=".dvc/cache", state_file="cache_state.json"):
    """Track changes in the .dvc/cache directory and return the list of changed files."""
    current_state = {}
    changed_files = []

    # Load the previous state if it exists
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            previous_state = json.load(f)
    else:
        previous_state = {}

    # Get the current state of the cache directory
    for root, _, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_hash = get_file_hash(file_path)
            current_state[file_path] = file_hash

            # Check if the file is new or has changed
            if file_path not in previous_state or previous_state[file_path] != file_hash:
                changed_files.append(file_path)

    # Save the current state to the state file
    with open(state_file, 'w') as f:
        json.dump(current_state, f)

    return changed_files

#uploading the results to the cloud using async 
async def upload(upload_file_queue, stop_event=None, demo_file="demo"):
    #build upload index from the cmf logging upload queue
    #what to be included in the upload index:
    # new file in the .dvc/cache
    # upload_queue: tar.gz and tar.gz.dvc
    #for model and mode.dvc
    # pipeline mlmd 
    # tar everything in the index and upload

    while not (stop_event and stop_event.is_set()):
        await asyncio.sleep(30)
        print("Checking for files to upload...")

        # Track file changes in the .dvc/cache directory
        changed_files = track_file_changes()
        if changed_files:
            print(f"Changed files: {changed_files}")
            for file in changed_files:
                await upload_file_queue.put(file)
        else:
            print("No new files to upload.")
        
        # Add demo file to the upload queue
        await upload_file_queue.put(demo_file)
        print(f"Added mlmd file to queue: {demo_file}")

        # Add cache_state.json to the upload queue
        cache_state_file = "cache_state.json"
        await upload_file_queue.put(cache_state_file)
        print(f"Added cache_state.json to queue: {cache_state_file}")

        # Add cmf_artifacts folder to the upload queue
        cmf_artifacts_folder = "cmf_artifacts"
        if os.path.exists(cmf_artifacts_folder):
            for root, _, files in os.walk(cmf_artifacts_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    await upload_file_queue.put(file_path)
                    print(f"Added {file_path} to queue")

        # Check if there are files to upload
        if upload_file_queue.empty():
            print("No files to upload.")
            continue

        #tar the files in cmf_upload queue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tar_path = f"upload_{timestamp}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            while not upload_file_queue.empty():
                file_path = await upload_file_queue.get()
                print(f"Archiving {file_path}")
                if os.path.isfile(file_path):
                    tar.add(file_path, arcname=os.path.basename(file_path))
                upload_file_queue.task_done()

        # Ensure the tar file is properly closed before uploading
        tar.close()

        # Upload the tar file to the remote host using SSH
        remote_host = ""
        remote_path = "/home/ubuntu/cmf-dev/cmf_proxy_demo"
        ssh_key_path = "/home/ubuntu/.ssh/id_rsa"

        async with asyncssh.connect(remote_host, username="ubuntu", client_keys=[ssh_key_path]) as conn:
            await asyncssh.scp(tar_path, (conn, remote_path))
            print(f"Uploaded {tar_path} to {remote_host}:{remote_path}")

async def tasks(video_path, output_dir, archive_dir, model_path, metawriter, demo_file):
    result_queue = asyncio.Queue()
    cmf_logging_queue = asyncio.Queue()
    upload_queue = asyncio.Queue()

    tasks = [
        asyncio.create_task(process_video(video_path, output_dir=output_dir, result_queue=result_queue)),
        asyncio.create_task(archive_and_log_results(output_dir=archive_dir, result_queue=result_queue, cmf_queue=cmf_logging_queue, upload_queue=upload_queue,metawriter=metawriter, model_path=model_path)),
        asyncio.create_task(upload(upload_queue, demo_file=demo_file))
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Call the ini.sh script
    try:
        subprocess.run(["/bin/bash", "init.sh"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running init.sh: {e}")
        exit(1)
   
    model_dir = "models"
    model_name = 'yolov8n.pt'
    model_path = os.path.join(model_dir, model_name)
    model = get_model(model_dir, model_name)

    video_dir = "videos"
    video_name = "input_video.mp4"
    video_path = os.path.join(video_dir, video_name)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    pipeline_name="demo"
    # Create a new context and execution for the pipeline
    metawriter = cmf.Cmf(pipeline_name, pipeline_name)
    stage_name = "inference"
    execution_name = "inference"
    _ = metawriter.create_context(pipeline_stage=str(stage_name))
    _ = metawriter.create_execution(execution_type=str(execution_name))
   
    demo_file = pipeline_name  # Specify the path to the demo file

    loop = asyncio.get_event_loop()
    loop.run_until_complete(tasks(video_path, output_dir, archive_dir, model_path, metawriter, demo_file))




