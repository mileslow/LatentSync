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
import argparse
import sys
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
# Note: DETECT_RESULTS_DIR is deprecated - each evaluation creates its own unique directory

# Global model instances (loaded once at startup)
syncnet = None


def initialize_models():
    """Initialize SyncNet models on CPU"""
    global syncnet
    
    device = "cpu"  # Force CPU usage
    print(f"[INFO] Initializing models on {device}...")
    
    # Initialize SyncNet evaluation model
    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(SYNCNET_MODEL_PATH)
    print("[INFO] SyncNet model loaded successfully")
    
    # Note: SyncNetDetector instances are created per-evaluation to avoid contamination


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
        # Check if file exists first
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file does not exist: {video_path}")
            return 0
        
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
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffprobe failed: {e.stderr if e.stderr else str(e)}")
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to get video duration: {str(e)}")
        return 0


def preprocess_video(input_path: str, output_path: str, start_time: float = None) -> bool:
    """
    Preprocess video: trim to 5 seconds at specified time, convert to 25fps
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the processed video
        start_time: Start time in seconds (if None, uses default strategy)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get video duration
        duration = get_video_duration(input_path)
        if duration == 0:
            print("[ERROR] Could not determine video duration")
            return False
        
        # Determine clip duration and start time
        clip_duration = min(5, duration)
        
        if start_time is None:
            # Default: 5-second interval before the last 10 seconds
            end_time = duration - 10
            start_time = max(0, end_time - 5)
            print(f"[INFO] Using default strategy: 5s before last 10s")
        else:
            # Use provided start time
            start_time = max(0, min(start_time, duration - clip_duration))
        
        print(f"[INFO] Extracting {clip_duration:.1f}s clip from {start_time:.2f}s to {start_time + clip_duration:.2f}s")
        
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


def evaluate_sync(video_path: str, temp_dir: str, detect_results_dir: str = None) -> dict:
    """
    Evaluate audio-visual synchronization of a video
    
    Args:
        video_path: Path to the video file
        temp_dir: Temporary directory for processing
        detect_results_dir: Directory for face detection results (optional, will create unique one if not provided)
        
    Returns:
        Dictionary with sync results
    """
    try:
        print(f"[INFO] Starting sync evaluation for: {video_path}")
        
        # Create a unique detect_results_dir for this evaluation to avoid contamination
        if detect_results_dir is None:
            detect_results_dir = os.path.join(temp_dir, f"detect_results_{uuid.uuid4().hex[:8]}")
        
        # Create a detector instance with unique directory
        local_detector = SyncNetDetector(device="cpu", detect_results_dir=detect_results_dir)
        
        # Run SyncNet detection
        print("[INFO] Running face detection and tracking...")
        local_detector(video_path=video_path, min_track=50)
        print("[INFO] Face detection completed")
        
        # Check if faces were detected
        crop_dir = os.path.join(detect_results_dir, "crop")
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
            # Use a separate temp directory for syncnet to avoid it deleting our files
            syncnet_temp_dir = os.path.join(temp_dir, f"syncnet_temp_{i}")
            av_offset, _, conf = syncnet.evaluate(
                video_path=crop_video_path,
                temp_dir=syncnet_temp_dir
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


def run_local_mode(video_path: str) -> None:
    """
    Run sync detection on a local video file (CLI mode)
    
    Args:
        video_path: Path to the local video file
    """
    print("=" * 60)
    print("Audio-Visual Sync Detection - Local Mode")
    print("=" * 60)
    
    # Validate input file
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.isfile(video_path):
        print(f"[ERROR] Path is not a file: {video_path}")
        sys.exit(1)
    
    print(f"\n[INFO] Processing local video: {video_path}")
    
    # Initialize models
    print("\n[INFO] Initializing models...")
    initialize_models()
    
    # Create temporary directory for processing
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join(TEMP_DIR_BASE, request_id)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[INFO] Created temporary directory: {temp_dir}")
    
    try:
        # Get video duration
        duration = get_video_duration(video_path)
        if duration == 0:
            print("[ERROR] Could not determine video duration")
            sys.exit(1)
        
        print(f"[INFO] Video duration: {duration:.1f}s")
        
        # Define sampling strategies
        sampling_strategies = []
        
        # Strategy 1: 5s before last 10s (default)
        if duration >= 15:
            sampling_strategies.append({
                "name": "end section (5s before last 10s)",
                "start_time": duration - 15
            })
        
        # Strategy 2: Middle 5s
        if duration >= 5:
            middle_start = max(0, (duration - 5) / 2)
            sampling_strategies.append({
                "name": "middle section",
                "start_time": middle_start
            })
        
        # Strategy 3: First 5s
        sampling_strategies.append({
            "name": "beginning section",
            "start_time": 0
        })
        
        # Strategy 4: 25% through the video
        if duration >= 20:
            quarter_start = duration * 0.25
            sampling_strategies.append({
                "name": "quarter section (25% through)",
                "start_time": quarter_start
            })
        
        print(f"[INFO] Will try {len(sampling_strategies)} sampling location(s)\n")
        
        # Try each sampling strategy
        all_results = []
        
        for i, strategy in enumerate(sampling_strategies, 1):
            print(f"\n[ATTEMPT {i}/{len(sampling_strategies)}] Trying {strategy['name']}...")
            
            # Create processed video path
            processed_video_path = os.path.join(temp_dir, f"processed_video_{i}.mp4")
            
            # Preprocess video
            if not preprocess_video(video_path, processed_video_path, start_time=strategy['start_time']):
                print(f"[WARN] Preprocessing failed for {strategy['name']}, trying next...")
                continue
            
            # Evaluate sync
            try:
                result = evaluate_sync(processed_video_path, temp_dir)
                
                if result["success"] and result.get("num_faces_detected", 0) > 0:
                    result_copy = result.copy()
                    result_copy["location"] = strategy["name"]
                    all_results.append(result_copy)
                    
                    if result.get("is_synced", False):
                        print(f"[SUCCESS] Found SYNCED faces in {strategy['name']} - confidence: {result.get('confidence', 0):.2f}")
                    else:
                        print(f"[WARN] Found faces in {strategy['name']} but DESYNCED (confidence: {result.get('confidence', 0):.2f})")
                else:
                    print(f"[WARN] No faces detected in {strategy['name']}")
                    
            except Exception as e:
                print(f"[ERROR] Exception during evaluation of {strategy['name']}: {str(e)}")
                continue
        
        # Print final results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        
        # Calculate statistics based on ALL attempted segments
        total_attempts = len(sampling_strategies)
        successful_detections = len(all_results)
        no_face_segments = total_attempts - successful_detections
        
        if len(all_results) == 0:
            print("\n[RESULT] No faces detected in any location")
            print("[STATUS] SYNCED (indeterminate - cannot determine sync status)")
            print("\nNote: No faces were detected in the video, so sync status")
            print("cannot be determined. The video is classified as synced by default.")
        else:
            # Count desynced segments (segments with faces that are desynced)
            desynced_count = sum(1 for r in all_results if not r.get("is_synced", False))
            synced_count = len(all_results) - desynced_count
            
            # Calculate desync percentage based on TOTAL attempts (not just successful detections)
            # Segments with no faces are treated as "not desynced" (neutral/synced)
            desync_percentage = (desynced_count / total_attempts) * 100
            
            # Determine final classification (75% threshold)
            is_desynced = desync_percentage >= 75.0
            
            # Get best result
            best_result = max(all_results, key=lambda x: x.get('confidence', 0))
            
            print(f"\nTotal segments sampled: {total_attempts}")
            print(f"  - Segments with faces detected: {successful_detections}")
            print(f"    • Synced: {synced_count}")
            print(f"    • Desynced: {desynced_count}")
            print(f"  - Segments with no faces: {no_face_segments}")
            print(f"\nDesync percentage: {desync_percentage:.1f}% ({desynced_count}/{total_attempts} segments desynced)")
            print(f"Best confidence: {best_result.get('confidence', 0):.2f}")
            print(f"Best AV offset: {best_result.get('av_offset', 'N/A')}")
            print(f"Threshold for sync: {SYNC_CONFIDENCE_THRESHOLD}")
            print(f"Classification threshold: 75% of total segments must be desynced")
            
            print(f"\n[STATUS] {'DESYNCED' if is_desynced else 'SYNCED'}")
            
            if is_desynced:
                print(f"\nThe video is classified as DESYNCED because {desync_percentage:.1f}%")
                print(f"of ALL sampled locations were desynced ({desynced_count}/{total_attempts} >= 75% threshold).")
            else:
                print(f"\nThe video is classified as SYNCED because {desync_percentage:.1f}%")
                print(f"of ALL sampled locations were desynced ({desynced_count}/{total_attempts} < 75% threshold).")
            
            # Show per-location details
            print("\nPer-location details:")
            for i, r in enumerate(all_results, 1):
                status = "SYNCED" if r.get("is_synced", False) else "DESYNCED"
                print(f"  {i}. {r.get('location', 'unknown')}: {status} (confidence: {r.get('confidence', 0):.2f})")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"\n[INFO] Temporary files saved in: {temp_dir}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": syncnet is not None,
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
            
            # Get video duration to plan sampling strategies
            duration = get_video_duration(raw_video_path)
            if duration == 0:
                print("[ERROR] Could not determine video duration")
                return jsonify({
                    "success": False,
                    "error": "Could not determine video duration"
                }), 400
            
            # Define multiple sampling strategies (5-second clips at different positions)
            sampling_strategies = []
            
            # Strategy 1: 5s before last 10s (default)
            if duration >= 15:
                sampling_strategies.append({
                    "name": "end section (5s before last 10s)",
                    "start_time": duration - 15
                })
            
            # Strategy 2: Middle 5s
            if duration >= 5:
                middle_start = max(0, (duration - 5) / 2)
                sampling_strategies.append({
                    "name": "middle section",
                    "start_time": middle_start
                })
            
            # Strategy 3: First 5s
            sampling_strategies.append({
                "name": "beginning section",
                "start_time": 0
            })
            
            # Strategy 4: 25% through the video
            if duration >= 20:
                quarter_start = duration * 0.25
                sampling_strategies.append({
                    "name": "quarter section (25% through)",
                    "start_time": quarter_start
                })
            
            print(f"[INFO] Video duration: {duration:.1f}s, will try {len(sampling_strategies)} sampling location(s)")
            
            # Try each sampling strategy and collect all results
            result = None
            all_results = []  # Track all successful detections
            errors_encountered = []  # Track errors during processing
            successful_attempts = 0
            
            for i, strategy in enumerate(sampling_strategies, 1):
                print(f"\n[ATTEMPT {i}/{len(sampling_strategies)}] Trying {strategy['name']}...")
                
                # Verify raw video still exists
                if not os.path.exists(raw_video_path):
                    print(f"[ERROR] Raw video file missing: {raw_video_path}")
                    errors_encountered.append({
                        "location": strategy["name"],
                        "error": "Raw video file missing"
                    })
                    continue
                
                # Create unique processed video path for this attempt
                processed_video_path = os.path.join(temp_dir, f"processed_video_{i}.mp4")
                
                # Preprocess video with this strategy
                if not preprocess_video(raw_video_path, processed_video_path, start_time=strategy['start_time']):
                    print(f"[WARN] Preprocessing failed for {strategy['name']}, trying next...")
                    errors_encountered.append({
                        "location": strategy["name"],
                        "error": "Preprocessing failed"
                    })
                    continue
                
                # Evaluate sync on preprocessed video (wrap in try-except to catch errors)
                try:
                    print(f"[INFO] Starting sync evaluation for {strategy['name']}...")
                    result = evaluate_sync(processed_video_path, temp_dir)
                    
                    # Check if we successfully detected faces
                    if result["success"] and result.get("num_faces_detected", 0) > 0:
                        successful_attempts += 1
                        result_copy = result.copy()
                        result_copy["location"] = strategy["name"]
                        all_results.append(result_copy)
                        
                        if result.get("is_synced", False):
                            print(f"[SUCCESS] Found SYNCED faces in {strategy['name']} - confidence: {result.get('confidence', 0):.2f}")
                        else:
                            print(f"[WARN] Found faces in {strategy['name']} but DESYNCED (confidence: {result.get('confidence', 0):.2f})")
                        
                        # Continue trying other locations to get more data points
                        continue
                    else:
                        print(f"[WARN] No faces detected in {strategy['name']}, trying next location...")
                        # Continue to next location (no cleanup)
                        continue
                        
                except Exception as e:
                    # Track the error
                    error_msg = str(e)
                    print(f"[ERROR] Exception during evaluation of {strategy['name']}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    errors_encountered.append({
                        "location": strategy["name"],
                        "error": error_msg
                    })
                    continue
            
            # Check if we had errors during processing
            if errors_encountered:
                if successful_attempts > 0:
                    # We had some successes but also errors - this indicates a problem
                    error_summary = "; ".join([f"{e['location']}: {e['error']}" for e in errors_encountered])
                    print(f"[ERROR] Mixed results - {successful_attempts} successful, {len(errors_encountered)} errors")
                    print(f"[ERROR] Errors: {error_summary}")
                    return jsonify({
                        "success": False,
                        "error": f"Processing failed for some locations: {error_summary}",
                        "successful_attempts": successful_attempts,
                        "failed_attempts": len(errors_encountered),
                        "errors": errors_encountered
                    }), 500
                elif len(errors_encountered) == len(sampling_strategies):
                    # ALL attempts failed with errors
                    error_summary = "; ".join([f"{e['location']}: {e['error']}" for e in errors_encountered])
                    print(f"[ERROR] All {len(sampling_strategies)} attempts failed with errors")
                    print(f"[ERROR] Errors: {error_summary}")
                    return jsonify({
                        "success": False,
                        "error": f"All processing attempts failed: {error_summary}",
                        "failed_attempts": len(errors_encountered),
                        "errors": errors_encountered
                    }), 500
            
            # Determine final result based on majority vote (75% threshold)
            # Calculate based on TOTAL attempts, not just successful detections
            total_attempts = len(sampling_strategies)
            
            if len(all_results) == 0:
                # No faces found in any location - classify as synced (cannot determine desync without faces)
                print("[FINAL] No faces detected in any location - classifying as SYNCED (indeterminate)")
                result = {
                    "success": True,
                    "confidence": 0.0,
                    "av_offset": None,
                    "is_synced": True,
                    "threshold": SYNC_CONFIDENCE_THRESHOLD,
                    "num_faces_detected": 0,
                    "all_confidences": [],
                    "attempts": total_attempts,
                    "successful_detections": 0,
                    "note": "No faces detected in any sampled location - cannot determine sync status"
                }
            else:
                # Count desynced segments (segments with faces that are desynced)
                desynced_count = sum(1 for r in all_results if not r.get("is_synced", False))
                synced_count = len(all_results) - desynced_count
                
                # Calculate desync percentage based on TOTAL attempts (not just successful detections)
                # Segments with no faces are treated as "not desynced" (neutral/synced)
                desync_percentage = (desynced_count / total_attempts) * 100
                
                # Only classify as desynced if 75% or more of ALL attempts are desynced
                is_desynced = desync_percentage >= 75.0
                
                # Get the best result (highest confidence) for reporting
                best_result = max(all_results, key=lambda x: x.get('confidence', 0))
                location = best_result.pop("location", "unknown")
                
                print(f"[FINAL] {len(all_results)} successful detection(s): {synced_count} synced, {desynced_count} desynced")
                print(f"[FINAL] Desync percentage: {desync_percentage:.1f}% ({desynced_count}/{total_attempts} total segments)")
                print(f"[FINAL] Threshold: 75% of total segments must be desynced to classify as DESYNCED")
                
                if is_desynced:
                    print(f"[FINAL] {desync_percentage:.1f}% desynced >= 75% - classifying as DESYNCED")
                    result = best_result
                    result["is_synced"] = False
                    result["attempts"] = total_attempts
                    result["successful_detections"] = len(all_results)
                    result["desync_percentage"] = round(desync_percentage, 1)
                else:
                    print(f"[FINAL] {desync_percentage:.1f}% desynced < 75% - classifying as SYNCED")
                    result = best_result
                    result["is_synced"] = True
                    result["attempts"] = total_attempts
                    result["successful_detections"] = len(all_results)
                    result["desync_percentage"] = round(desync_percentage, 1)
            
            # No cleanup - keep all files for debugging
            
            if result["success"]:
                print(f"[SUCCESS] Request completed successfully - Synced: {result['is_synced']}\n")
                return jsonify(result), 200
            else:
                print(f"[ERROR] Request failed: {result.get('error', 'Unknown error')}\n")
                return jsonify(result), 500
                
        except Exception as e:
            # Log error without cleanup
            print(f"[ERROR] Exception during processing: {str(e)}")
            import traceback
            traceback.print_exc()
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
            "Samples 5-second clips from multiple locations in the video",
            "Processing time depends on video length"
        ]
    })


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Audio-Visual Sync Detection - API Server or Local Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API server (default)
  python main.py
  
  # Check sync status of a local video file
  python main.py --local /path/to/video.mp4
  python main.py -l /path/to/video.mp4
        """
    )
    parser.add_argument(
        '--local', '-l',
        type=str,
        metavar='VIDEO_PATH',
        help='Run in local mode with a video file path (instead of API server)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=int(os.environ.get('PORT', 8080)),
        help='Port for API server (default: 8080 or PORT env var)'
    )
    
    args = parser.parse_args()
    
    # Run in appropriate mode
    if args.local:
        # Local mode - process single video file
        run_local_mode(args.local)
    else:
        # API server mode
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
            port=args.port,
            debug=False,  # Set to False in production
            threaded=True
        )

