import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from .box_utils import nms_


class S3FD:

    def __init__(self, device="cpu"):

        tstamp = time.time()
        self.device = device

        print("[S3FD] loading YOLOv8n on", self.device)
        self.net = YOLO("yolov8n.pt")
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
                        classes = result.boxes.cls.cpu().numpy()
                        
                        # Filter for person class (0) in COCO
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
