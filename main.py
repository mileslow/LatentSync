"""
Flask API for Audio-Visual Sync Detection
Classifies whether video audio is in sync or desynced (CPU-only)
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path
from statistics import fmean
import subprocess
import json
import logging
import cv2
from flask import Flask, request, jsonify
import requests
import torch
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from ultralytics import YOLO

from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from latentsync.utils.util import red_text

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
SYNCNET_MODEL_PATH = "checkpoints/auxiliary/syncnet_v2.model"
SYNC_CONFIDENCE_THRESHOLD = 3.0  # Threshold for binary classification
TEMP_DIR_BASE = "temp_api"
DETECT_RESULTS_DIR = "detect_results_api"

# Global model instances (loaded once at startup)
syncnet = None
syncnet_detector = None
yolo_face_detector = None


def initialize_models():
    """Initialize SyncNet models and YOLO on CPU"""
    global syncnet, syncnet_detector, yolo_face_detector
    
    device = "cpu"  # Force CPU usage
    logger.info(f"Initializing models on {device}...")
    
    # Initialize SyncNet evaluation model
    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(SYNCNET_MODEL_PATH)
    logger.info("SyncNet model loaded successfully")
    
    # Initialize SyncNet detector (face detection + processing)
    syncnet_detector = SyncNetDetector(device=device, detect_results_dir=DETECT_RESULTS_DIR)
    logger.info("SyncNet detector loaded successfully")
    
    # Initialize YOLO for preprocessing scene selection
    logger.info("Loading YOLOv8n for scene selection...")
    yolo_face_detector = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")


def download_video(video_url: str, output_path: str) -> bool:
    """
    Download video from URL to local path
    
    Args:
        video_url: URL of the video to download
        output_path: Local path to save the video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Video downloaded successfully to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        return False


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Duration in seconds, or 0 if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        logger.info(f"Video duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        logger.error(f"Failed to get video duration: {str(e)}")
        return 0


def check_scene_has_face(video_path: str, start_time: float, end_time: float, yolo_model: YOLO) -> bool:
    """
    Check if a scene has detectable faces using YOLO
    
    Args:
        video_path: Path to the video file
        start_time: Scene start time in seconds
        end_time: Scene end time in seconds
        yolo_model: YOLO face detection model
        
    Returns:
        True if faces detected, False otherwise
    """
    try:
        # Sample a few frames from the scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Check frames at 25%, 50%, 75% of the scene
        sample_times = [
            start_time + (end_time - start_time) * 0.25,
            start_time + (end_time - start_time) * 0.50,
            start_time + (end_time - start_time) * 0.75,
        ]
        
        for sample_time in sample_times:
            frame_num = int(sample_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run YOLO detection
                results = yolo_model(frame_rgb, conf=0.3, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Check if any person/face detected
                    classes = results[0].boxes.cls.cpu().numpy()
                    if 0 in classes:  # Person class
                        cap.release()
                        return True
        
        cap.release()
        return False
        
    except Exception as e:
        logger.error(f"Error checking scene for faces: {str(e)}")
        return False


def preprocess_video(input_path: str, output_path: str, yolo_model: YOLO) -> bool:
    """
    Preprocess video: Find 20-second segment with faces, convert to 25fps
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the processed video
        yolo_model: YOLO model for face detection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get video duration
        duration = get_video_duration(input_path)
        if duration == 0:
            logger.error("Could not determine video duration")
            return False
        
        # Start with 40-second interval before the last 10 seconds
        initial_end = duration - 10
        initial_start = max(0, initial_end - 40)
        
        logger.info(f"Initial search range: {initial_start:.2f}s - {initial_end:.2f}s (40s window)")
        
        # Run scene detection on the initial 40-second interval
        logger.info("Running scene detection on initial 40s interval...")
        scene_list = detect(
            input_path, 
            ContentDetector(),
            start_time=initial_start,
            end_time=initial_end
        )
        
        if not scene_list:
            logger.warning("No scenes detected in initial interval")
            # Check if the initial interval has faces
            available_duration = initial_end - initial_start
            if check_scene_has_face(input_path, initial_start, initial_end, yolo_model):
                logger.info(f"Faces found in initial interval, using {min(20, available_duration):.1f}s segment")
                start_time = initial_start
                clip_duration = min(20, available_duration)
            else:
                logger.warning(f"No faces in initial interval, using {min(20, available_duration):.1f}s anyway")
                start_time = initial_start
                clip_duration = min(20, available_duration)
        else:
            logger.info(f"Detected {len(scene_list)} scene(s) in initial interval")
            
            # Collect all scenes
            all_scenes = []
            for scene in scene_list:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                all_scenes.append({
                    'start': start_sec,
                    'end': end_sec,
                    'duration': end_sec - start_sec
                })
            
            # Sort scenes by start time (newest first, working backwards)
            all_scenes.sort(key=lambda x: x['start'], reverse=True)
            
            # Find a segment (up to 20s) with faces
            # Strategy: Check each scene, if it has faces, use up to 20s starting from that scene
            selected_start = None
            clip_duration = None
            
            for i, scene in enumerate(all_scenes):
                logger.info(f"Checking scene {i+1}/{len(all_scenes)}: {scene['start']:.2f}s - {scene['end']:.2f}s ({scene['duration']:.2f}s)")
                
                # Check if this scene has faces
                if check_scene_has_face(input_path, scene['start'], scene['end'], yolo_model):
                    logger.info(f"Faces detected in scene {i+1}")
                    
                    # Try to use 20 seconds starting from this scene
                    potential_start = scene['start']
                    potential_end = min(potential_start + 20, initial_end)
                    available_duration = potential_end - potential_start
                    
                    # Accept if we have at least 15 seconds
                    if available_duration >= 15:
                        logger.info(f"Using {available_duration:.1f}s segment starting from scene {i+1}: {potential_start:.2f}s - {potential_end:.2f}s")
                        selected_start = potential_start
                        clip_duration = available_duration
                        break
                    else:
                        logger.info(f"Only {available_duration:.1f}s available from this scene (too short), checking next...")
                else:
                    logger.info(f"No faces in scene {i+1}, checking next...")
            
            if selected_start is not None:
                start_time = selected_start
                # clip_duration already set in the loop
            else:
                logger.warning("No suitable scenes with faces found, using fallback")
                # Fallback: use up to 20s from the beginning of initial interval
                start_time = initial_start
                clip_duration = min(20, initial_end - initial_start)
        
        # Trim and reencode the selected portion
        cmd = [
            'ffmpeg',
            '-threads', '0',
            '-y',
            '-nostdin',
            '-loglevel', 'error',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(clip_duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-r', '25',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-ar', '16000',
            '-ac', '1',
            '-b:a', '128k',
            '-async', '1',
            '-vsync', 'cfr',
            output_path
        ]
        
        logger.info(f"Trimming and encoding: {start_time:.2f}s - {start_time + clip_duration:.2f}s ({clip_duration:.2f}s) at 25fps")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Video preprocessed successfully: {output_path}")
            return True
        else:
            logger.error("Preprocessed video is empty or not created")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg preprocessing failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Video preprocessing failed: {str(e)}")
        return False


def evaluate_sync(video_path: str, temp_dir: str) -> dict:
    """
    Evaluate audio-visual synchronization of a video
    
    Args:
        video_path: Path to the video file
        temp_dir: Temporary directory for processing
        
    Returns:
        Dictionary with sync results
    """
    try:
        logger.info(f"Starting sync evaluation for: {video_path}")
        
        # Run SyncNet detection
        logger.info("Running face detection and tracking...")
        syncnet_detector(video_path=video_path, min_track=50)
        logger.info("Face detection completed")
        
        # Check if faces were detected
        crop_dir = os.path.join(DETECT_RESULTS_DIR, "crop")
        crop_videos = os.listdir(crop_dir) if os.path.exists(crop_dir) else []
        
        logger.info(f"Found {len(crop_videos)} face track(s)")
        
        if not crop_videos:
            logger.info("No faces detected in the video - assuming synced")
            return {
                "success": True,
                "confidence": None,
                "av_offset": None,
                "is_synced": True,
                "threshold": SYNC_CONFIDENCE_THRESHOLD,
                "num_faces_detected": 0,
                "all_confidences": [],
                "note": "No faces detected - defaulting to synced"
            }
        
        # Evaluate sync for each detected face
        results = []
        
        for i, video in enumerate(crop_videos, 1):
            logger.info(f"Evaluating face track {i}/{len(crop_videos)}: {video}")
            crop_video_path = os.path.join(crop_dir, video)
            av_offset, _, conf = syncnet.evaluate(
                video_path=crop_video_path,
                temp_dir=temp_dir
            )
            results.append({'offset': av_offset, 'confidence': conf})
            logger.info(f"Face {i} - Offset: {av_offset}, Confidence: {conf:.2f}")
        
        # Use MAXIMUM confidence (best synced face)
        # This correctly handles videos where only some people are speaking
        best_result = max(results, key=lambda x: x['confidence'])
        max_confidence = best_result['confidence']
        best_offset = best_result['offset']
        
        # Binary classification: synced or desynced based on best face
        is_synced = max_confidence >= SYNC_CONFIDENCE_THRESHOLD
        
        all_confidences = [r['confidence'] for r in results]
        logger.info(f"Best confidence: {max_confidence:.2f} (from track with offset: {best_offset})")
        logger.info(f"All confidences: {all_confidences}")
        logger.info(f"Result: {'SYNCED' if is_synced else 'DESYNCED'}")
        
        return {
            "success": True,
            "confidence": round(max_confidence, 2),
            "av_offset": best_offset,
            "is_synced": is_synced,
            "threshold": SYNC_CONFIDENCE_THRESHOLD,
            "num_faces_detected": len(crop_videos),
            "all_confidences": [round(r['confidence'], 2) for r in results]
        }
        
    except Exception as e:
        logger.error(f"Sync evaluation failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "confidence": None,
            "av_offset": None,
            "is_synced": None
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": syncnet is not None and syncnet_detector is not None,
        "device": "cpu"
    })


@app.route('/detect_sync', methods=['POST'])
def detect_sync():
    """
    Main endpoint for sync detection
    
    Expected JSON body:
    {
        "video_url": "https://example.com/video.mp4"
    }
    
    Returns:
    {
        "success": true,
        "is_synced": true/false,
        "confidence": 5.23,
        "av_offset": 0,
        "threshold": 3.0,
        "num_faces_detected": 1
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'video_url' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'video_url' in request body"
            }), 400
        
        video_url = data['video_url']
        logger.info(f"New sync detection request for: {video_url}")
        
        # Create unique temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = os.path.join(TEMP_DIR_BASE, request_id)
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_dir}")
        
        raw_video_path = os.path.join(temp_dir, "raw_video.mp4")
        processed_video_path = os.path.join(temp_dir, "processed_video.mp4")
        
        try:
            # Download video
            if not download_video(video_url, raw_video_path):
                logger.error("Video download failed")
                return jsonify({
                    "success": False,
                    "error": "Failed to download video from URL"
                }), 400
            
            # Check if file exists and has content
            file_size = os.path.getsize(raw_video_path) if os.path.exists(raw_video_path) else 0
            logger.info(f"Raw video file size: {file_size / (1024*1024):.2f} MB")
            
            if not os.path.exists(raw_video_path) or file_size == 0:
                logger.error("Downloaded video is empty or invalid")
                return jsonify({
                    "success": False,
                    "error": "Downloaded video file is empty or invalid"
                }), 400
            
            # Preprocess video: find scene with faces, trim, and convert to 25fps
            logger.info("Preprocessing video (scene detection and trimming)...")
            if not preprocess_video(raw_video_path, processed_video_path, yolo_face_detector):
                logger.error("Video preprocessing failed")
                return jsonify({
                    "success": False,
                    "error": "Failed to preprocess video"
                }), 400
            
            # Evaluate sync on preprocessed video
            logger.info("Starting sync evaluation pipeline...")
            result = evaluate_sync(processed_video_path, temp_dir)
            
            # Clean up temporary files
            logger.info("Cleaning up temporary files...")
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temp directory: {temp_dir}")
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
                logger.info(f"Removed detect results directory: {DETECT_RESULTS_DIR}")
            
            if result["success"]:
                logger.info(f"Request completed successfully - Synced: {result['is_synced']}")
                return jsonify(result), 200
            else:
                logger.error(f"Request failed: {result.get('error', 'Unknown error')}")
                return jsonify(result), 500
                
        except Exception as e:
            # Clean up on error
            logger.error(f"Exception during processing: {str(e)}", exc_info=True)
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
            raise e
            
    except Exception as e:
        logger.error(f"Unhandled exception in detect_sync: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "service": "Audio-Visual Sync Detection API",
        "version": "1.0",
        "device": "cpu",
        "endpoints": {
            "/health": "GET - Health check",
            "/detect_sync": "POST - Detect audio-visual sync from video URL",
            "/": "GET - This documentation"
        },
        "usage": {
            "endpoint": "/detect_sync",
            "method": "POST",
            "body": {
                "video_url": "https://example.com/video.mp4"
            },
            "response": {
                "success": True,
                "is_synced": True,
                "confidence": 5.23,
                "av_offset": 0,
                "threshold": 3.0,
                "num_faces_detected": 1
            }
        },
        "notes": [
            "Confidence >= 3.0 indicates the video is in sync",
            "AV offset is measured in frames (25 FPS)",
            "Requires faces to be visible in the video",
            "Processing time depends on video length"
        ]
    })


if __name__ == '__main__':
    # Initialize models before starting server
    print("=" * 60)
    print("Audio-Visual Sync Detection API (CPU-only)")
    print("=" * 60)
    
    initialize_models()
    
    print("=" * 60)
    print("Starting Flask server...")
    print("=" * 60)
    
    # Get port from environment variable (for Cloud Run compatibility)
    port = int(os.environ.get('PORT', 8080))
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to False in production
        threaded=True
    )

