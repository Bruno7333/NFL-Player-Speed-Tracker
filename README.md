# NFL Player Speed Tracker

A computer vision pipeline that detects NFL field markings from broadcast video, maps pixel coordinates to real-world field positions in yards, tracks individual players using YOLO + ByteTrack, and calculates their speed in mph — all in real time.

---

## How It Works

The pipeline runs in three stages per frame:

```
Video Frame
    │
    ▼
Phase 1 — Field Calibration
  • HSV white mask + green proximity filter isolates field lines
  • Canny edge detection + HoughLines finds candidate lines
  • Lines are grouped and classified as yard lines or sidelines
  • A vanishing point is computed from the yard lines
  • One detected sideline + vanishing point builds a homography matrix H
  • H maps any pixel (x, y) → real field coordinates (yards along, yards across)
    │
    ▼
Phase 2 — Player Detection & Tracking
  • YOLOv8 (pre-trained on COCO) detects all people in the frame
  • ByteTrack assigns persistent IDs across frames
  • Click on a player to lock onto them — their bounding box and foot point are tracked
  • Foot point (bottom-center of bounding box) is mapped to field coords via H
    │
    ▼
Phase 3 — Speed Calculation
  • Field positions are buffered over a rolling window of frames
  • Distance (yards) ÷ time (seconds) × 2.04545 = speed in mph
  • A median filter smooths out noise; readings above 25 mph are discarded
    │
    ▼
Overlay: bounding box, speed label, field coords, bird's-eye minimap
```

---

## Requirements

**Python 3.10+** is recommended.

### Install dependencies

```bash
pip install opencv-python numpy ultralytics matplotlib
```

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video I/O, image processing, HoughLines, homography |
| `numpy` | Array math throughout the pipeline |
| `ultralytics` | YOLOv8 model + ByteTrack tracker |
| `matplotlib` | Debug heatmap visualization (testing mode) |

On first run, `ultralytics` will automatically download `yolov8m.pt` (~50 MB) if it is not already present.

---

## Getting a Video Clip

### Supported formats
Any format OpenCV can read: `.mp4`, `.mkv`, `.webm`, `.avi`, etc.

### Camera angle requirement
**Only use clips filmed from a standard broadcast sideline camera angle** — the wide shot where you can see the full width of the field and multiple yard lines in perspective. The homography math relies on being able to detect at least 2 yard lines and 1 sideline.

![Example of a valid camera angle](https://static.www.nfl.com/image/upload/t_editorial_landscape_mobile/f_auto/league/wlqagcngd21zxmwv6ur4.jpg)

Clips that will **not** work well:
- End zone / red zone overhead cameras
- Close-up player cameras
- All-22 film (top-down angle)
- Heavy zoom shots where fewer than 2 yard lines are visible

### Downloading clips with yt-dlp

Install yt-dlp:
```bash
pip install yt-dlp
```

Download a specific clip (YouTube, Twitter/X, etc.):
```bash
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "YOUR_URL_HERE" -o "Images/clip.mp4"
```

Download just the video (no audio, smaller file):
```bash
yt-dlp -f "bestvideo[ext=mp4]" "YOUR_URL_HERE" -o "Images/clip.mp4"
```

Trim to a specific time range (requires `ffmpeg`):
```bash
yt-dlp --download-sections "*0:10-0:40" "YOUR_URL_HERE" -o "Images/clip.mp4"
```

Once downloaded, update `VIDEO_PATH` in `main.py`:
```python
VIDEO_PATH = 'Images/clip.mp4'
```

---

## Running the Desktop Viewer

```bash
python main.py
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `e` | Step forward one frame (while paused) |
| `q` | Step backward one frame (while paused) |
| `c` | Clear current player selection |
| `d` | Exit |

### Mouse
**Left-click** on any player to begin tracking them. Their bounding box, field position, and speed will appear. Grey dots mark all other detected players.

---

## Configuration

All tunable constants live at the top of `main.py`.

### TESTING_MODE

```python
TESTING_MODE = True   # default
```

When `True`, the viewer opens three extra debug windows:

- **NFL Live** — main frame with tab20 colour-coded lines (each detected line group gets a unique colour) and field overlay
- **NFL Edges** — the Canny edge map fed into HoughLines
- **NFL B&W** — the white mask after green proximity filtering
- **Rho/Theta Heatmap** — a 2D scatter plot of every detected line in (theta°, rho px) space. Each cluster of dots represents a distinct line on the field. Yard lines cluster near theta ≈ 0°; sidelines cluster near theta ≈ ±90°. The dashed white line is the first-pass fit through yard-line candidates; the solid yellow line is the refined fit after outlier removal. Use this window to diagnose why lines are being misclassified or grouped incorrectly.

When `False`, debug windows are hidden, yard lines draw in red, and sidelines draw in blue.

### CROP_BOTTOM

```python
CROP_BOTTOM = 0   # default (disabled)
```

Some broadcast clips have a **scoreboard bar** along the bottom of the frame. The white text and graphics in that bar can be misdetected as yard lines, confusing the homography.

Set `CROP_BOTTOM` to the height in pixels of the scoreboard bar to blank it out before edge detection:

```python
CROP_BOTTOM = 60   # mask the bottom 60 rows
```

To find the right value: run with `TESTING_MODE = True`, watch the **Rho/Theta Heatmap** and **NFL B&W** windows, and increase `CROP_BOTTOM` until the spurious horizontal clusters disappear from the heatmap.

---

## Project Structure

```
main.py               Desktop viewer — full pipeline with OpenCV windows
field_calibration.py  Phase 1: line detection, homography, minimap utilities
player_tracking.py    Phase 2: foot-point extraction utility
Images/               Put your video clips here
```
