# 🎥 Smart Camera Recorder API

A powerful Flask-based video recording and retrieval system with:

- Live USB camera stream
- Continuous high-FPS recording (segment-based)
- Efficient memory buffer for frame-level retrieval
- Timestamp-based video clip extraction (in-memory or via file slicing)
- Auto-cleanup of old recordings
- RESTful endpoints to control and access everything

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`cv2`)
- FFmpeg installed (`ffmpeg`, `ffprobe`)
- USB camera

Install Python dependencies:

```
pip install -r requirements.txt
```

Install FFmpeg (Ubuntu):

```
sudo apt install ffmpeg
```

---

## 🧠 Features

- Continuous camera recording in 10-minute `.mp4` segments
- Automatic cleanup of recordings older than 72 hours
- Optional in-memory frame buffer (~30 mins) for fast video extraction
- FFmpeg-based clip generation from timestamp range
- Live video feed via `/`
- REST API for camera info, recordings, and footage download

---

## 📦 API Endpoints

### `GET /`
Live MJPEG stream from the camera.

---

### `GET /camera_info`
Returns available camera IDs, resolution, FPS, and backend info.

---

### `GET /recordings`
Lists all recorded video segments.

**Response JSON:**
```json
[
  {
    "file": "recording_20240517_103000.mp4",
    "start_time": "20240517_103000",
    "end_time": "20240517_104000",
    "size": 12432545,
    "duration": 600.0
  },
  ...
]
```

---

### `GET /footage?from=YYYYMMDD_HHMMSS&to=YYYYMMDD_HHMMSS`

Returns a video clip from the requested timestamp range (via:
- 🧠 in-memory frame buffer (if recent)
- 📂 disk recordings using `ffmpeg` (if older)

**Example:**
```
/footage?from=20240517_102300&to=20240517_102318
```

Returns an `.mp4` clip covering the time range.

---

## 🧪 Development

Run the Flask app:

```
python app.py
```

Live stream:  
[http://localhost:5000/](http://localhost:5000/)

Recording clips:
- Are saved in `/recordings`
- Index is saved in `recordings/recording_index.json`
- Extracted clips are saved in `/clips`

---

## 🗃️ Directory Structure

```
recordings/          # Stores continuous video segments
clips/               # Stores extracted clips
recording_index.json # Tracks segment metadata
```

---

## 🛠️ Notes

- This system supports 60 FPS continuous recording
- Default recording codec is MJPG, fallback to raw if needed
- Automatically falls back to available camera if configured one is missing
- Memory buffer is capped to ~30 minutes of frame history

---

## 🚀 Deployment Tips

- Use `gunicorn` with `threaded=True` for better performance
- Mount volumes for `/recordings` and `/clips` if deploying in Docker
- Customize FPS, retention duration, and segment length as needed
