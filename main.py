"""
Flask API for Audio-Visual Sync Detection
Classifies whether video audio is in sync or desynced (CPU-only)
"""

import os
import tempfile
import uuid
from pathlib import Path
from statistics import fmean
import subprocess
import json
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
        print(f"[INFO] Video duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        print(f"[ERROR] Failed to get video duration: {str(e)}")
        return 0


def preprocess_video(input_path: str, output_path: str) -> bool:
    """
    Preprocess video: trim to 40 seconds before the last 10 seconds, convert to 25fps
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the processed video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get video duration
        duration = get_video_duration(input_path)
        if duration == 0:
            print("[ERROR] Could not determine video duration")
            return False
        
        # Calculate 40-second interval before the last 10 seconds
        # End time: duration - 10 seconds
        # Start time: end_time - 40 seconds
        end_time = duration - 10
        start_time = max(0, end_time - 40)
        
        # Adjust if video is shorter than 50 seconds
        if duration < 50:
            print(f"[WARN] Video is only {duration:.2f}s, using first 40s or entire video")
            start_time = 0
            clip_duration = min(40, duration)
        else:
            clip_duration = 40
            print(f"[INFO] Extracting 40s clip from {start_time:.2f}s to {end_time:.2f}s")
        
        # Use ffmpeg to trim and reencode
        # -threads 0: auto-detect optimal thread count
        # -ss: start time (placed before -i for faster seeking)
        # -t: duration
        # Force reencode with proper settings for sync detection
        cmd = [
            'ffmpeg',
            '-threads', '0',
            '-y',  # overwrite output
            '-nostdin',
            '-loglevel', 'error',
            '-ss', str(start_time),
            '-i', input_path,
            '-t', str(clip_duration),
            '-c:v', 'libx264',  # Force H.264 video reencoding
            '-preset', 'medium',  # Encoding speed/quality tradeoff
            '-crf', '23',  # Constant quality (lower = better quality)
            '-r', '25',  # Force 25 fps output
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-c:a', 'aac',  # Force AAC audio reencoding
            '-ar', '16000',  # Audio sample rate (16kHz for sync detection)
            '-ac', '1',  # Mono audio
            '-b:a', '128k',  # Audio bitrate
            '-async', '1',  # Audio sync method
            '-vsync', 'cfr',  # Constant frame rate
            output_path
        ]
        
        print(f"[INFO] Preprocessing video: {start_time:.2f}s - {start_time + clip_duration:.2f}s at 25fps")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"[INFO] Video preprocessed successfully: {output_path}")
            return True
        else:
            print("[ERROR] Preprocessed video is empty or not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg preprocessing failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Video preprocessing failed: {str(e)}")
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
        print(f"[INFO] Starting sync evaluation for: {video_path}")
        
        # Run SyncNet detection
        print("[INFO] Running face detection and tracking...")
        syncnet_detector(video_path=video_path, min_track=50)
        print("[INFO] Face detection completed")
        
        # Check if faces were detected
        crop_dir = os.path.join(DETECT_RESULTS_DIR, "crop")
        crop_videos = os.listdir(crop_dir) if os.path.exists(crop_dir) else []
        
        print(f"[INFO] Found {len(crop_videos)} face track(s)")
        
        if not crop_videos:
            print("[WARN] No faces detected in the video")
            return {
                "success": False,
                "error": "No faces detected in the video",
                "confidence": None,
                "av_offset": None,
                "is_synced": None
            }
        
        # Evaluate sync for each detected face
        results = []
        
        for i, video in enumerate(crop_videos, 1):
            print(f"[INFO] Evaluating face track {i}/{len(crop_videos)}: {video}")
            crop_video_path = os.path.join(crop_dir, video)
            av_offset, _, conf = syncnet.evaluate(
                video_path=crop_video_path,
                temp_dir=temp_dir
            )
            results.append({'offset': av_offset, 'confidence': conf})
            print(f"[INFO] Face {i} - Offset: {av_offset}, Confidence: {conf:.2f}")
        
        # Use MAXIMUM confidence (best synced face)
        # This correctly handles videos where only some people are speaking
        best_result = max(results, key=lambda x: x['confidence'])
        max_confidence = best_result['confidence']
        best_offset = best_result['offset']
        
        # Binary classification: synced or desynced based on best face
        is_synced = max_confidence >= SYNC_CONFIDENCE_THRESHOLD
        
        all_confidences = [r['confidence'] for r in results]
        print(f"[INFO] Best confidence: {max_confidence:.2f} (from track with offset: {best_offset})")
        print(f"[INFO] All confidences: {all_confidences}")
        print(f"[INFO] Result: {'SYNCED' if is_synced else 'DESYNCED'}")
        
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
        print(f"[ERROR] Sync evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print(f"\n[REQUEST] New sync detection request for: {video_url}")
        
        # Create unique temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = os.path.join(TEMP_DIR_BASE, request_id)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[INFO] Created temporary directory: {temp_dir}")
        
        raw_video_path = os.path.join(temp_dir, "raw_video.mp4")
        processed_video_path = os.path.join(temp_dir, "processed_video.mp4")
        
        try:
            # Download video
            if not download_video(video_url, raw_video_path):
                print("[ERROR] Video download failed")
                return jsonify({
                    "success": False,
                    "error": "Failed to download video from URL"
                }), 400
            
            # Check if file exists and has content
            file_size = os.path.getsize(raw_video_path) if os.path.exists(raw_video_path) else 0
            print(f"[INFO] Raw video file size: {file_size / (1024*1024):.2f} MB")
            
            if not os.path.exists(raw_video_path) or file_size == 0:
                print("[ERROR] Downloaded video is empty or invalid")
                return jsonify({
                    "success": False,
                    "error": "Downloaded video file is empty or invalid"
                }), 400
            
            # Preprocess video: trim to 20s and convert to 25fps
            print("[INFO] Preprocessing video (trimming and converting to 25fps)...")
            if not preprocess_video(raw_video_path, processed_video_path):
                print("[ERROR] Video preprocessing failed")
                return jsonify({
                    "success": False,
                    "error": "Failed to preprocess video"
                }), 400
            
            # Evaluate sync on preprocessed video
            print("[INFO] Starting sync evaluation pipeline...")
            result = evaluate_sync(processed_video_path, temp_dir)
            
            # Clean up temporary files
            print("[INFO] Cleaning up temporary files...")
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[INFO] Removed temp directory: {temp_dir}")
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
                print(f"[INFO] Removed detect results directory: {DETECT_RESULTS_DIR}")
            
            if result["success"]:
                print(f"[SUCCESS] Request completed successfully - Synced: {result['is_synced']}\n")
                return jsonify(result), 200
            else:
                print(f"[ERROR] Request failed: {result.get('error', 'Unknown error')}\n")
                return jsonify(result), 500
                
        except Exception as e:
            # Clean up on error
            print(f"[ERROR] Exception during processing: {str(e)}")
            import shutil
            import traceback
            traceback.print_exc()
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(DETECT_RESULTS_DIR):
                shutil.rmtree(DETECT_RESULTS_DIR)
            raise e
            
    except Exception as e:
        print(f"[ERROR] Unhandled exception in detect_sync: {str(e)}")
        import traceback
        traceback.print_exc()
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

