from flask import Flask, Response, request, send_file, jsonify
import cv2
import threading
import time
import os
import queue
import subprocess
from datetime import datetime, timedelta
import glob
import numpy as np
import json
import shutil

app = Flask(__name__)

# Camera Settings
CAMERA_ID = 2
FPS = 60  # Keep your high FPS since resources aren't an issue
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Create a dedicated directory for clips
CLIPS_DIR = "clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

# Index file to track all recordings and their timestamps
INDEX_FILE = f"{RECORDINGS_DIR}/recording_index.json"

# Frame queue for streaming
frame_queue = queue.Queue(maxsize=30)  # Larger buffer for high FPS

# Store frames in memory for quick retrieval
frame_buffer = {}
buffer_lock = threading.Lock()
buffer_duration = 1800  # 30 minutes of memory buffer with your resources

# Recording file tracking with metadata
recording_index = {}


# --- Recording Index Functions ---
def load_recording_index():
    global recording_index
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r') as f:
                recording_index = json.load(f)
                print(f"Loaded recording index with {len(recording_index)} entries")
        except Exception as e:
            print(f"Error loading recording index: {e}")
            recording_index = {}
    else:
        recording_index = {}


def save_recording_index():
    try:
        with open(INDEX_FILE, 'w') as f:
            json.dump(recording_index, f)
        print(f"Saved recording index with {len(recording_index)} entries")
    except Exception as e:
        print(f"Error saving recording index: {e}")


def update_recording_index(file_path, start_time, end_time):
    global recording_index
    key = os.path.basename(file_path)
    recording_index[key] = {
        "file_path": file_path,
        "start_time": start_time,
        "end_time": end_time,
        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    # Save the index after each update for reliability
    save_recording_index()


def find_recordings_for_timerange(from_ts, to_ts):
    """Find all recording files that cover the requested time range"""
    from_dt = datetime.strptime(from_ts, "%Y%m%d_%H%M%S")
    to_dt = datetime.strptime(to_ts, "%Y%m%d_%H%M%S")

    matching_files = []

    for filename, metadata in recording_index.items():
        start_time = datetime.strptime(metadata["start_time"], "%Y%m%d_%H%M%S")
        end_time = datetime.strptime(metadata["end_time"], "%Y%m%d_%H%M%S")

        # If this recording overlaps with our requested range
        if (start_time <= to_dt and end_time >= from_dt):
            matching_files.append(metadata["file_path"])

    # Sort chronologically
    matching_files.sort()

    return matching_files


# --- Find available cameras ---
def list_available_cameras():
    available_cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()

    if available_cameras:
        print(f"Available cameras: {available_cameras}")
    else:
        print("No cameras found!")

    return available_cameras


# --- Detect available codecs ---
def find_working_codec():
    # Try different codecs in order of preference
    codecs = [
        ('MJPG', 'MJPG'),  # Motion JPEG
        ('XVID', 'XVID'),  # XVID
        ('MP4V', 'MP4V'),  # MPEG-4
        ('X264', 'X264'),  # X264
        ('H264', 'H264')  # H264
    ]

    # Create a small test frame
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    test_path = f"{RECORDINGS_DIR}/codec_test.avi"

    for codec_name, codec_str in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            out = cv2.VideoWriter(test_path, fourcc, 30, (100, 100))
            if out.isOpened():
                out.write(test_frame)
                out.release()
                if os.path.exists(test_path):
                    os.remove(test_path)
                print(f"Using codec: {codec_name} ({codec_str})")
                return fourcc, codec_name
        except Exception as e:
            print(f"Codec {codec_name} failed: {e}")

    # Fallback to uncompressed
    print("All codecs failed, using raw uncompressed video (large files!)")
    return 0, "RAW"


# --- Continuous recording thread ---
def record_video():
    global frame_buffer, recording_index

    # Load existing recording index
    load_recording_index()

    # First, list available cameras
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("[ERROR] No cameras found. Please check your USB camera connection.")
        return

    camera_id = CAMERA_ID
    if camera_id not in available_cameras:
        if available_cameras:
            camera_id = available_cameras[0]
            print(f"Requested camera ID {CAMERA_ID} not available. Using camera ID {camera_id} instead.")
        else:
            print("[ERROR] No cameras available.")
            return

    # Find a working codec
    CODEC_FOURCC, CODEC_NAME = find_working_codec()

    # Open the camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_id}. Please check your USB camera connection.")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FPS, FPS)  # Set FPS

    # Try to set some camera parameters that might help with USB cameras
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG if available
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Larger buffer with your resources
    except:
        pass

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened with resolution: {width}x{height}, FPS: {actual_fps}")
    print(f"Camera backend: {cap.getBackendName()}")

    # Create a new writer every 10 minutes
    out = None
    last_writer_time = time.time()
    current_video_path = ""
    segment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_frame_time = time.time()
    frames_count = 0

    while True:
        try:
            # Create a new video file every 10 minutes
            current_time = time.time()
            if out is None or (current_time - last_writer_time) > 600:  # 10 minutes
                if out is not None:
                    out.release()
                    # Record the end time of the segment
                    segment_end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Update index with completed recording
                    update_recording_index(current_video_path, segment_start_time, segment_end_time)

                # Start a new segment
                segment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_video_path = f"{RECORDINGS_DIR}/recording_{segment_start_time}.mp4"

                # Try creating a video writer with the selected codec
                out = cv2.VideoWriter(current_video_path, CODEC_FOURCC, FPS, (width, height))

                if not out.isOpened():
                    print(f"Failed to open video writer with codec {CODEC_NAME}, falling back to raw")
                    out = cv2.VideoWriter(current_video_path, 0, FPS, (width, height))

                last_writer_time = current_time
                print(f"Created new video file: {current_video_path} with codec {CODEC_NAME}")

                # Clean up old recordings - keep only the latest 72 hours with your resources
                # This is done by checking file creation dates in the index
                now = datetime.now()
                old_files_to_remove = []

                for filename, metadata in recording_index.items():
                    try:
                        file_time = datetime.strptime(metadata["created_at"], "%Y%m%d_%H%M%S")
                        if (now - file_time).total_seconds() > 72 * 3600:  # 72 hours
                            old_files_to_remove.append(filename)
                    except Exception as e:
                        print(f"Error checking file age: {e}")

                for filename in old_files_to_remove:
                    try:
                        file_path = recording_index[filename]["file_path"]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Removed old recording: {file_path}")
                        del recording_index[filename]
                    except Exception as e:
                        print(f"Error removing old file: {e}")

                # Save updated index
                save_recording_index()

            # Read frame with timeout
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame from camera")
                time.sleep(0.01)  # Small sleep to prevent CPU hogging on failure

                # Try to reconnect if no frames for 5 seconds
                if time.time() - last_frame_time > 5:
                    print("No frames for 5 seconds, trying to reconnect camera...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_id)
                    # Reset the last frame time to avoid immediate reconnection
                    last_frame_time = time.time()

                continue

            # Reset last frame time on successful read
            last_frame_time = time.time()

            # Record frame
            out.write(frame)

            # Get current timestamp with millisecond precision
            current_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Store frame in memory buffer with timestamp
            with buffer_lock:
                frame_buffer[current_ts] = frame.copy()

                # Clean old frames from buffer
                now = time.time()
                current_dt = datetime.now()
                old_keys = []

                for k in list(frame_buffer.keys()):
                    try:
                        frame_dt = datetime.strptime(k, "%Y%m%d_%H%M%S_%f")
                        age = (current_dt - frame_dt).total_seconds()
                        if age > buffer_duration:
                            old_keys.append(k)
                    except ValueError:
                        old_keys.append(k)

                for k in old_keys:
                    del frame_buffer[k]

            # Update queue for streaming (non-blocking)
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                # If queue is full, remove oldest frame and add new one
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass

            # Calculate and print actual FPS every second
            frames_count += 1
            if time.time() - last_frame_time >= 1.0:
                print(
                    f"Recording FPS: {frames_count}, Buffer size: {len(frame_buffer)}, Recordings: {len(recording_index)}")
                frames_count = 0
                last_frame_time = time.time()

        except Exception as e:
            print(f"Error in recording thread: {e}")
            time.sleep(1)  # Sleep on error to prevent tight loops


# --- Camera information endpoint ---
@app.route('/camera_info')
def camera_info():
    available_cameras = list_available_cameras()

    # Try to get detailed info about the camera
    info = {"available_cameras": available_cameras}

    if available_cameras:
        try:
            cap = cv2.VideoCapture(CAMERA_ID)
            if cap.isOpened():
                info["current_camera"] = CAMERA_ID
                info["resolution"] = {
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
                info["fps"] = cap.get(cv2.CAP_PROP_FPS)
                info["backend"] = cap.getBackendName()
                cap.release()
        except Exception as e:
            info["error"] = str(e)

    return jsonify(info)


# --- Live video stream ---
@app.route('/')
def live():
    def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=0.5)
                # Use lower quality for streaming to improve performance
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except queue.Empty:
                # If no new frame, yield empty content to keep connection alive
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- List all recordings ---
@app.route('/recordings')
def list_recordings():
    # Return the complete recording index, sorted by timestamp
    recordings = []

    for filename, metadata in recording_index.items():
        if os.path.exists(metadata["file_path"]):
            recordings.append({
                'file': os.path.basename(metadata["file_path"]),
                'start_time': metadata["start_time"],
                'end_time': metadata["end_time"],
                'size': metadata["file_size"],
                'duration': (datetime.strptime(metadata["end_time"], "%Y%m%d_%H%M%S") -
                             datetime.strptime(metadata["start_time"], "%Y%m%d_%H%M%S")).total_seconds()
            })

    # Sort by start time
    recordings.sort(key=lambda x: x['start_time'], reverse=True)

    return jsonify(recordings)


# --- IMPROVED: Get video clip from timestamps ---
@app.route('/footage')
def footage():
    from_ts = request.args.get("from")  # Format: YYYYMMDD_HHMMSS
    to_ts = request.args.get("to")

    if not from_ts or not to_ts:
        return "Usage: /footage?from=YYYYMMDD_HHMMSS&to=YYYYMMDD_HHMMSS", 400

    fmt = "%Y%m%d_%H%M%S"
    try:
        from_dt = datetime.strptime(from_ts, fmt)
        to_dt = datetime.strptime(to_ts, fmt)
    except ValueError:
        return "Invalid timestamp format", 400

    duration = (to_dt - from_dt).total_seconds()
    if duration <= 0:
        return "Invalid time range", 400

    # Output clip path - always use mp4 for compatibility
    clip_name = f"clip_{from_ts}_{to_ts}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_name)

    # Check if we already have this clip cached
    if os.path.exists(clip_path):
        print(f"Using cached clip: {clip_path}")
        return send_file(clip_path, mimetype='video/mp4', as_attachment=False)

    print(f"Creating new clip for time range: {from_ts} to {to_ts}")

    # Try both methods in parallel for reliability
    memory_success = False
    file_success = False

    # If duration is short and we have the frames in memory, create clip directly
    if duration <= buffer_duration:
        try:
            # Find frames in the requested time range
            from_ts_full = from_ts + "_000"  # Add milliseconds
            to_ts_full = to_ts + "_999"  # Add milliseconds

            with buffer_lock:
                # Get relevant frames
                frames = []
                for ts, frame in frame_buffer.items():
                    if from_ts_full <= ts <= to_ts_full:
                        frames.append((ts, frame))

                # Sort frames by timestamp
                frames.sort(key=lambda x: x[0])
                frames = [frame for _, frame in frames]

            if frames:
                # Create a temp directory for frames
                temp_dir = f"{CLIPS_DIR}/temp_{from_ts}_{to_ts}"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)  # Clean up any existing temp dir
                os.makedirs(temp_dir, exist_ok=True)

                # Save frames as JPG images
                for i, frame in enumerate(frames):
                    jpg_path = f"{temp_dir}/frame_{i:06d}.jpg"
                    cv2.imwrite(jpg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Use ffmpeg to create video from images with high quality settings
                memory_clip_path = f"{temp_dir}/memory_clip.mp4"
                command = [
                    "ffmpeg", "-y",
                    "-framerate", str(FPS),
                    "-i", f"{temp_dir}/frame_%06d.jpg",
                    "-c:v", "libx264",
                    "-preset", "medium",  # Better quality, you have resources
                    "-crf", "18",  # High quality
                    memory_clip_path
                ]

                process = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if process.returncode == 0 and os.path.exists(memory_clip_path):
                    # Copy to final destination
                    shutil.copy2(memory_clip_path, clip_path)
                    memory_success = True
                    print(f"Created clip from memory: {clip_path} with {len(frames)} frames")
                else:
                    print(f"FFmpeg error: {process.stderr}")

                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning temp dir: {e}")
        except Exception as e:
            print(f"Error creating clip from memory: {e}")

    # If memory method didn't work or footage is older, use recording files
    if not memory_success:
        try:
            # Find all recording files that might contain our time range
            matching_files = find_recordings_for_timerange(from_ts, to_ts)

            if not matching_files:
                return "No recording files available for the requested time range", 404

            print(f"Found {len(matching_files)} recording files for time range")

            # If we have multiple files, we need to create a file list for ffmpeg
            if len(matching_files) > 1:
                # Create a temp file list for ffmpeg
                file_list_path = f"{CLIPS_DIR}/filelist_{from_ts}_{to_ts}.txt"
                with open(file_list_path, 'w') as f:
                    for file_path in matching_files:
                        f.write(f"file '{os.path.abspath(file_path)}'\n")

                # Concatenate the files
                concat_path = f"{CLIPS_DIR}/concat_{from_ts}_{to_ts}.mp4"
                concat_command = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", file_list_path,
                    "-c", "copy",
                    concat_path
                ]

                process = subprocess.run(
                    concat_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if process.returncode != 0:
                    print(f"FFmpeg concat error: {process.stderr}")
                    # Fall back to just using the first file that might contain our time range
                    input_file = matching_files[0]
                else:
                    input_file = concat_path
            else:
                input_file = matching_files[0]

            # Convert from_ts to time offset within the file
            file_start = None
            if len(matching_files) == 1:
                # Find the start time of this file from the index
                file_name = os.path.basename(matching_files[0])
                if file_name in recording_index:
                    file_start = datetime.strptime(recording_index[file_name]["start_time"], "%Y%m%d_%H%M%S")

            if file_start:
                # Calculate time offset from file start
                offset_seconds = (from_dt - file_start).total_seconds()
                if offset_seconds < 0:
                    offset_seconds = 0
                start_time = str(timedelta(seconds=offset_seconds))
            else:
                # Use absolute timestamp as best effort
                start_time = from_dt.strftime('%H:%M:%S')

            # Cut video using ffmpeg with optimized parameters
            extract_command = [
                "ffmpeg", "-y",
                "-ss", start_time,
                "-i", input_file,
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "medium",  # Better quality with your resources
                "-crf", "18",  # High quality
                clip_path
            ]

            process = subprocess.run(
                extract_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode == 0 and os.path.exists(clip_path):
                file_success = True
                print(f"Successfully created clip from files: {clip_path}")

                # Clean up temp files
                if len(matching_files) > 1:
                    if os.path.exists(file_list_path):
                        os.remove(file_list_path)
                    if os.path.exists(concat_path):
                        os.remove(concat_path)
            else:
                print(f"FFmpeg error: {process.stderr}")

        except Exception as e:
            print(f"Exception in footage file extraction: {e}")

    # Return the clip if either method succeeded
    if memory_success or file_success:
        if os.path.exists(clip_path):
            # Check if the file is actually valid
            check_command = ["ffprobe", "-v", "error", clip_path]
            process = subprocess.run(
                check_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode == 0:
                # File is valid, return it
                return send_file(clip_path, mimetype='video/mp4', as_attachment=False)
            else:
                return f"Error: Created clip is invalid. Please try again.", 500
        else:
            return "Error: Failed to create video clip", 500
    else:
        return "Error: Could not create clip using either method. Check logs for details.", 500


if __name__ == '__main__':
    # Start recording thread
    threading.Thread(target=record_video, daemon=True).start()
    # Run Flask with threaded=True for better performance
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)