"""
Flask API for Audio-Visual Sync Detection
Classifies whether video audio is in sync or desynced (CPU-only)
"""

import os
import tempfile
import uuid
from pathlib import Path
from statistics import fmean
from flask import Flask, request, jsonify
import requests
import torch

from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from latentsync.utils.util import red_text

app = Flask(__name__)

# Configuration
SYNCNET_MODEL_PATH = "checkpoints/auxiliary/syncnet_v2.model"
SYNC_CONFIDENCE_THRESHOLD = 3.0  # Threshold for binary classification
TEMP_DIR_BASE = "temp_api"
DETECT_RESULTS_DIR = "detect_results_api"

# Global model instances (loaded once at startup)
syncnet = None
syncnet_detector = None


def initialize_models():
    """Initialize SyncNet models on CPU"""
    global syncnet, syncnet_detector
    
    device = "cpu"  # Force CPU usage
    print(f"[INFO] Initializing models on {device}...")
    
    # Initialize SyncNet evaluation model
    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(SYNCNET_MODEL_PATH)
    print("[INFO] SyncNet model loaded successfully")
    
    # Initialize SyncNet detector (face detection + processing)
    syncnet_detector = SyncNetDetector(device=device, detect_results_dir=DETECT_RESULTS_DIR)
    print("[INFO] SyncNet detector loaded successfully")


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
        print(f"[INFO] Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"[INFO] Video downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download video: {str(e)}")
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
        # Run SyncNet detection
        syncnet_detector(video_path=video_path, min_track=50)
        
        # Check if faces were detected
        crop_dir = os.path.join(DETECT_RESULTS_DIR, "crop")
        crop_videos = os.listdir(crop_dir) if os.path.exists(crop_dir) else []
        
        if not crop_videos:
            return {
                "success": False,
                "error": "No faces detected in the video",
                "confidence": None,
                "av_offset": None,
                "is_synced": None
            }
        
        # Evaluate sync for each detected face
        av_offset_list = []
        conf_list = []
        
        for video in crop_videos:
            crop_video_path = os.path.join(crop_dir, video)
            av_offset, _, conf = syncnet.evaluate(
                video_path=crop_video_path,
                temp_dir=temp_dir
            )
            av_offset_list.append(av_offset)
            conf_list.append(conf)
        
        # Calculate average confidence and offset
        avg_confidence = fmean(conf_list)
        avg_offset = int(fmean(av_offset_list))
        
        # Binary classification: synced or desynced
        is_synced = avg_confidence >= SYNC_CONFIDENCE_THRESHOLD
        
        return {
            "success": True,
            "confidence": round(avg_confidence, 2),
            "av_offset": avg_offset,
            "is_synced": is_synced,
            "threshold": SYNC_CONFIDENCE_THRESHOLD,
            "num_faces_detected": len(crop_videos)
        }
        
    except Exception as e:
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
        
        # Create unique temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = os.path.join(TEMP_DIR_BASE, request_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        video_path = os.path.join(temp_dir, "input_video.mp4")
        
        try:
            # Download video
            if not download_video(video_url, video_path):
                return jsonify({
                    "success": False,
                    "error": "Failed to download video from URL"
                }), 400
            
            # Check if file exists and has content
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return jsonify({
                    "success": False,
                    "error": "Downloaded video file is empty or invalid"
                }), 400
            
            # Evaluate sync
            result = evaluate_sync(video_path, temp_dir)
            
            # Clean up temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
            
            if result["success"]:
                return jsonify(result), 200
            else:
                return jsonify(result), 500
                
        except Exception as e:
            # Clean up on error
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
            raise e
            
    except Exception as e:
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
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False in production
        threaded=True
    )

