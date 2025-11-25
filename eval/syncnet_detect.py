# Adapted from https://github.com/joonson/syncnet_python/blob/master/run_pipeline.py

import os, pdb, subprocess, glob, cv2
import numpy as np
from shutil import rmtree
import torch

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from eval.detectors import S3FD


class SyncNetDetector:
    def __init__(self, device, detect_results_dir="detect_results"):
        # S3FD now uses YOLOv8n under the hood - much lighter and faster
        self.face_detector = S3FD(device=device)
        self.detect_results_dir = detect_results_dir

    def __call__(self, video_path: str, min_track=50, scale=False):
        print(f"[SYNCNET] Starting face detection pipeline")
        print(f"[SYNCNET] Input video: {video_path}")
        print(f"[SYNCNET] Min track length: {min_track} frames")
        
        crop_dir = os.path.join(self.detect_results_dir, "crop")
        video_dir = os.path.join(self.detect_results_dir, "video")
        frames_dir = os.path.join(self.detect_results_dir, "frames")
        temp_dir = os.path.join(self.detect_results_dir, "temp")

        # ========== DELETE EXISTING DIRECTORIES ==========
        print("[SYNCNET] Cleaning up existing directories...")
        if os.path.exists(crop_dir):
            rmtree(crop_dir)

        if os.path.exists(video_dir):
            rmtree(video_dir)

        if os.path.exists(frames_dir):
            rmtree(frames_dir)

        if os.path.exists(temp_dir):
            rmtree(temp_dir)

        # ========== MAKE NEW DIRECTORIES ==========
        print("[SYNCNET] Creating working directories...")
        os.makedirs(crop_dir)
        os.makedirs(video_dir)
        os.makedirs(frames_dir)
        os.makedirs(temp_dir)

        # ========== PREPARE VIDEO ==========
        print("[SYNCNET] Processing video...")
        
        if scale:
            print("[SYNCNET] Scaling video to 224x224...")
            scaled_video_path = os.path.join(video_dir, "scaled.mp4")
            command = f"ffmpeg -threads 0 -loglevel error -y -nostdin -i {video_path} -vf scale='224:224' {scaled_video_path}"
            subprocess.run(command, shell=True)
            video_path = scaled_video_path
            print("[SYNCNET] Video scaling complete")
        
        # Copy video to working directory (already preprocessed at 25fps)
        print("[SYNCNET] Copying preprocessed video to working directory...")
        import shutil
        shutil.copy2(video_path, os.path.join(video_dir, 'video.mp4'))
        print("[SYNCNET] Video ready for processing")

        print("[SYNCNET] Extracting frames...")
        command = f"ffmpeg -threads 0 -y -nostdin -loglevel error -i {os.path.join(video_dir, 'video.mp4')} -qscale:v 2 -f image2 {os.path.join(frames_dir, '%06d.jpg')}"
        subprocess.run(command, shell=True, stdout=None)
        num_frames = len(glob.glob(os.path.join(frames_dir, "*.jpg")))
        print(f"[SYNCNET] Extracted {num_frames} frames")

        print("[SYNCNET] Extracting audio...")
        command = f"ffmpeg -threads 0 -y -nostdin -loglevel error -i {os.path.join(video_dir, 'video.mp4')} -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(video_dir, 'audio.wav')}"
        subprocess.run(command, shell=True, stdout=None)
        print("[SYNCNET] Audio extraction complete")

        print("[SYNCNET] Running face detection on all frames...")
        faces = self.detect_face(frames_dir)

        print("[SYNCNET] Detecting scene changes...")
        scene = self.scene_detect(video_dir)
        print(f"[SYNCNET] Found {len(scene)} scene(s)")

        # Face tracking
        print("[SYNCNET] Tracking faces across scenes...")
        alltracks = []

        for i, shot in enumerate(scene):
            shot_length = shot[1].frame_num - shot[0].frame_num
            if shot_length >= min_track:
                print(f"[SYNCNET] Scene {i+1}/{len(scene)}: {shot_length} frames, tracking faces...")
                tracks = self.track_face(faces[shot[0].frame_num : shot[1].frame_num], min_track=min_track)
                alltracks.extend(tracks)
                print(f"[SYNCNET] Found {len(tracks)} track(s) in scene {i+1}")
            else:
                print(f"[SYNCNET] Scene {i+1}/{len(scene)}: {shot_length} frames (too short, skipping)")

        print(f"[SYNCNET] Total face tracks found: {len(alltracks)}")

        # Face crop
        if len(alltracks) == 0:
            print("[SYNCNET] No face tracks found, skipping crop step")
        else:
            print(f"[SYNCNET] Cropping {len(alltracks)} face track(s)...")
            for ii, track in enumerate(alltracks):
                print(f"[SYNCNET] Cropping track {ii+1}/{len(alltracks)}...")
                self.crop_video(track, os.path.join(crop_dir, "%05d" % ii), frames_dir, 25, temp_dir, video_dir)
            print("[SYNCNET] All tracks cropped successfully")

        print("[SYNCNET] Cleaning up temporary files...")
        rmtree(temp_dir)
        print("[SYNCNET] Face detection pipeline complete")

    def scene_detect(self, video_dir):
        print("[SYNCNET] Initializing scene detection...")
        video_manager = VideoManager([os.path.join(video_dir, "video.mp4")])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        # Add ContentDetector algorithm (constructor takes detector options like threshold).
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()

        video_manager.set_downscale_factor()

        video_manager.start()

        print("[SYNCNET] Analyzing video for scene changes...")
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list(base_timecode)

        if scene_list == []:
            print("[SYNCNET] No scene changes detected, treating as single scene")
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
        else:
            print(f"[SYNCNET] Detected {len(scene_list)} scene(s)")

        return scene_list

    def track_face(self, scenefaces, num_failed_det=25, min_track=50, min_face_size=100):

        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []

        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if track == []:
                        track.append(face)
                        framefaces.remove(face)
                    elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                        iou = bounding_box_iou(face["bbox"], track[-1]["bbox"])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break

            if track == []:
                break
            elif len(track) > min_track:

                framenum = np.array([f["frame"] for f in track])
                bboxes = np.array([np.array(f["bbox"]) for f in track])

                frame_i = np.arange(framenum[0], framenum[-1] + 1)

                bboxes_i = []
                for ij in range(0, 4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)

                if (
                    max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
                    > min_face_size
                ):
                    tracks.append({"frame": frame_i, "bbox": bboxes_i})

        return tracks

    def detect_face(self, frames_dir, facedet_scale=0.25):
        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()
        total_frames = len(flist)
        print(f"[SYNCNET] Detecting faces in {total_frames} frames...")

        dets = []
        faces_found = 0
        progress_interval = max(1, total_frames // 10)  # Log every 10%

        for fidx, fname in enumerate(flist):
            if fidx % progress_interval == 0:
                progress = (fidx / total_frames) * 100
                print(f"[SYNCNET] Face detection progress: {progress:.0f}% ({fidx}/{total_frames} frames)")
            
            image = cv2.imread(fname)

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])

            dets.append([])
            for bbox in bboxes:
                dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]})
                faces_found += 1

        print(f"[SYNCNET] Face detection complete: {faces_found} face detection(s) across {total_frames} frames")
        return dets

    def crop_video(self, track, cropfile, frames_dir, frame_rate, temp_dir, video_dir, crop_scale=0.4):
        track_length = len(track["frame"])
        print(f"[SYNCNET]   Processing {track_length} frames for this track...")

        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vOut = cv2.VideoWriter(cropfile + "t.mp4", fourcc, frame_rate, (224, 224))

        dets = {"x": [], "y": [], "s": []}

        for det in track["bbox"]:

            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)  # crop center x
            dets["x"].append((det[0] + det[2]) / 2)  # crop center y

        # Smooth detections
        dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
        dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

        print(f"[SYNCNET]   Writing cropped video frames...")
        for fidx, frame in enumerate(track["frame"]):

            cs = crop_scale

            bs = dets["s"][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

            image = cv2.imread(flist[frame])

            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
            my = dets["y"][fidx] + bsi  # BBox center Y
            mx = dets["x"][fidx] + bsi  # BBox center X

            face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]

            vOut.write(cv2.resize(face, (224, 224)))

        audiotmp = os.path.join(temp_dir, "audio.wav")
        audiostart = (track["frame"][0]) / frame_rate
        audioend = (track["frame"][-1] + 1) / frame_rate

        vOut.release()

        # ========== CROP AUDIO FILE ==========
        print(f"[SYNCNET]   Cropping audio (%.2fs - %.2fs)..." % (audiostart, audioend))
        command = "ffmpeg -threads 0 -y -nostdin -loglevel error -i %s -ss %.3f -to %.3f %s" % (
            os.path.join(video_dir, "audio.wav"),
            audiostart,
            audioend,
            audiotmp,
        )
        output = subprocess.run(command, shell=True, stdout=None)

        sample_rate, audio = wavfile.read(audiotmp)

        # ========== COMBINE AUDIO AND VIDEO FILES ==========
        print(f"[SYNCNET]   Combining audio and video...")
        command = "ffmpeg -threads 0 -y -nostdin -loglevel error -i %st.mp4 -i %s -c:v copy -c:a aac %s.mp4" % (
            cropfile,
            audiotmp,
            cropfile,
        )
        output = subprocess.run(command, shell=True, stdout=None)

        os.remove(cropfile + "t.mp4")
        print(f"[SYNCNET]   Track crop complete")

        return {"track": track, "proc_track": dets}


def bounding_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
