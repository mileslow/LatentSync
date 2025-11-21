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

        print("[S3FD] loading YOLOv8n-face on", self.device)
        
        # Use YOLOv8 face detection model
        # Download from akanametov/yolov8-face repository
        model_path = "checkpoints/auxiliary/yolov8n-face.pt"
        
        if not os.path.exists(model_path):
            print("[S3FD] YOLOv8-face model not found, downloading...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download YOLOv8n face detection model
            model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
            
            try:
                response = requests.get(model_url, stream=True, timeout=120)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f"[S3FD] Downloaded YOLOv8-face model to {model_path}")
            except Exception as e:
                print(f"[S3FD] Failed to download face model: {e}")
                print("[S3FD] Falling back to base YOLOv8n")
                self.net = YOLO("yolov8n.pt")
                self.is_face_model = False
                print("[S3FD] finished loading (%.4f sec)" % (time.time() - tstamp))
                return
        
        self.net = YOLO(model_path)
        self.is_face_model = True
        print("[S3FD] finished loading (%.4f sec)" % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                # Lower conf threshold for better detection
                effective_conf = max(0.3, conf_th * 0.5)
                
                # Run YOLO inference
                results = self.net(image, conf=effective_conf, verbose=False, imgsz=int(640 * s), device=self.device)
                
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
