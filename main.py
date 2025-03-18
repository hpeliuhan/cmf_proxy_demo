import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
import asyncio

def get_model(model_path: str, model_name: str = 'yolov8n.pt'):
    full_model_path = os.path.join(model_path, model_name)
    
    if not os.path.exists(full_model_path):
        print(f"Model not found at {full_model_path}. Downloading...")
        os.makedirs(model_path, exist_ok=True)
        
        # Download the model
        model = YOLO(model_name)
        
        # Save the model to the specified path
        model.save(full_model_path)
        print(f"Model saved to {full_model_path}")

        #delete the model in the parent path
        os.remove(model_name)

    else:
        print(f"Model found at {full_model_path}. Loading...")

    # Load the model from local path
    return YOLO(full_model_path)

# Async function to save image and bounding boxes
async def save_frame(frame, boxes, frame_count,output_dir="output"):
    
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


async def process_video(video_path,output_dir="output"):
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

        # Save frame and bounding boxes asynchronously
        await save_frame(frame, boxes, frame_count,output_dir=output_dir)

        frame_count += 1
        time.sleep(1)
    
    cap.release()



if __name__=="__main__":
    model_dir = "models"
    model_name = 'yolov8n.pt'
    model = get_model(model_dir, model_name)

    video_dir="videos"
    video_name="input_video.mp4"

    output_dir="output"
    video_path = os.path.join(video_dir, video_name)
    
    asyncio.run(process_video(video_path,output_dir=output_dir))
   # process_video(video_path,output_dir=output_dir)
    


