import time
import numpy as np
import cv2
import torch
import os
from ultralytics import YOLO
from .box_utils import nms_
import requests


class S3FD:

    def __init__(self, device="cpu"):

        tstamp = time.time()
        self.device = device

        print(f"[S3FD] ========================================")
        print(f"[S3FD] Initializing S3FD detector")
        print(f"[S3FD] Device: {self.device}")
        print(f"[S3FD] Current working directory: {os.getcwd()}")
        
        # Use YOLOv8 face detection model
        # Download from akanametov/yolov8-face repository
        model_path = "checkpoints/auxiliary/yolov8n-face.pt"
        abs_model_path = os.path.abspath(model_path)
        
        print(f"[S3FD] Looking for model at: {abs_model_path}")
        print(f"[S3FD] Model exists: {os.path.exists(model_path)}")
        
        if os.path.exists(os.path.dirname(model_path)):
            files_in_dir = os.listdir(os.path.dirname(model_path))
            print(f"[S3FD] Files in {os.path.dirname(model_path)}: {files_in_dir}")
        else:
            print(f"[S3FD] Directory {os.path.dirname(model_path)} does not exist!")
        
        if not os.path.exists(model_path):
            print("[S3FD] YOLOv8-face model NOT found locally, attempting download...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download YOLOv8n face detection model
            model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
            print(f"[S3FD] Downloading from: {model_url}")
            
            try:
                response = requests.get(model_url, stream=True, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                print(f"[S3FD] Download size: {total_size / 1024 / 1024:.2f} MB")
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                print(f"[S3FD] Downloaded {downloaded / 1024 / 1024:.2f} MB to {model_path}")
                print(f"[S3FD] Model file size after download: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
            except Exception as e:
                print(f"[S3FD] FAILED to download face model: {type(e).__name__}: {e}")
                print("[S3FD] Falling back to base YOLOv8n (PERSON detector, not face-specific)")
                print("[S3FD] WARNING: This will result in worse sync detection accuracy!")
                self.net = YOLO("yolov8n.pt")
                self.is_face_model = False
                print("[S3FD] finished loading with FALLBACK model (%.4f sec)" % (time.time() - tstamp))
                print(f"[S3FD] ========================================")
                return
        else:
            print(f"[S3FD] Found cached model at: {model_path}")
            print(f"[S3FD] Model file size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
        
        print(f"[S3FD] Loading YOLOv8-face model from {model_path}...")
        self.net = YOLO(model_path)
        self.is_face_model = True
        print(f"[S3FD] Successfully loaded YOLOv8-FACE model (%.4f sec)" % (time.time() - tstamp))
        print(f"[S3FD] ========================================")

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                # Lower conf threshold for better detection
                effective_conf = max(0.3, conf_th * 0.5)
                
                # Run YOLO inference with deterministic settings
                results = self.net(image, conf=effective_conf, verbose=False, imgsz=int(640 * s), device=self.device, augment=False)
                
                if len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        
                        # For face-specific models, class 0 is "face"
                        # For base YOLO, class 0 is "person" 
                        # If using face model, no need to filter by class (all detections are faces)
                        if hasattr(self, 'is_face_model') and self.is_face_model:
                            # All detections are faces, no filtering needed
                            pass
                        else:
                            # Filter for person class (0) in COCO for base YOLO
                            classes = result.boxes.cls.cpu().numpy()
                            mask = classes == 0
                            boxes = boxes[mask]
                            confs = confs[mask]
                        
                        for box, conf in zip(boxes, confs):
                            bbox = (box[0], box[1], box[2], box[3], conf)
                            bboxes = np.vstack((bboxes, bbox))
            
            if len(bboxes) > 0:
                keep = nms_(bboxes, 0.1)
                bboxes = bboxes[keep]

        return bboxes
