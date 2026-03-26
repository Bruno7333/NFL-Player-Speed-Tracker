---
name: project-plan
description: >
  Master roadmap for building an NFL player speed tracking web application from the current
  OpenCV line-detection pipeline. Consult this skill whenever the user asks about the next
  implementation step, architecture decisions, model training, homography math, speed
  calculations, backend design, or web deployment for this project.
---

# NFL Player Speed Tracker — Full Project Plan

## Current State

**Active file:** `imageTestHL.py` — a full video pipeline (Phase 1 complete).

What it does per frame:
- Reads `Images/NFL05.webm` frame-by-frame via OpenCV (1280×720)
- HSV white mask (`lower=[0,0,130]`, `upper=[180,80,255]`) isolates field markings; scoreboard band (bottom 110 rows) zeroed before Canny
- `cv.HoughLines` → rho/theta normalised to (-π/2, π/2] for grouping
- Two-pass linear fit in (theta, rho) space identifies yard-line trend (`yl_fit2`)
- Temporal angle filter rejects noisy yard-line candidates across frames
- Groups merged in (theta, rho) space; classified by theta as sideline / yard_line / unknown
- Yard-line outliers filtered against `yl_fit2` residual (> 150 px → discarded)
- Sideline selected by **most-negative rho** (closest physical line to camera)
- **Hybrid VP homography**: one sideline + vanishing point of yard lines → synthesised second sideline; `cv.findHomography` → `H`
- `pixel_to_field(px, py, H)` maps any pixel to `(yard_along, yard_across)`
- Tab20 colour overlay + bird's-eye minimap (dynamic `field_length_visible`)
- Playback: Space = play/pause, e = next frame, q = prev frame, d = exit

---

## Phase 1 — Field Calibration & Homography ✅ COMPLETE

**Goal:** Map pixel coordinates → real-world field coordinates (yards).

### 1.1 Identify & Classify Reference Lines ✅
- HSV white mask → Canny → `cv.HoughLines`
- Rho/theta normalisation folds all lines to (-π/2, π/2] for consistent grouping
- Grouping in (theta, rho) space with `ANGLE_THRESHOLD=10°`, `RHO_THRESHOLD=30px`
- Theta-based classification: `|theta - π/2| < 20°` → sideline; `|theta| < 60°` → yard line
- Two-pass `yl_fit2` trend + temporal filter stabilise detection across frames
- Sideline selected by most-negative rho; yard lines sorted left→right and labeled 0, 5, 10, …

### 1.2 Compute Homography Matrix ✅ (Hybrid VP method)

**Key insight:** One sideline is reliably detected. A second sideline is synthesised using
the vanishing point of the yard lines, avoiding the need to detect it directly.

```python
# For each labeled yard line i:
P_i = intersect(primary_sideline, yard_line_i)      # → field (yard_num, 0.0)
t   = FIELD_WIDTH / (FIELD_WIDTH + VP_DEPTH_YARDS)  # ≈ 0.348
Q_i = P_i + t * (VP - P_i)                          # → field (yard_num, 53.3)

H, _ = cv.findHomography(src_pts, dst_pts)           # ≥ 4 point pairs required
```

Constants: `FIELD_WIDTH=53.3 yards`, `VP_DEPTH_YARDS=100.0` (tunable).
Requires ≥ 2 detected yard lines.

### 1.3 Field Coordinate Transformer ✅

```python
def pixel_to_field(px, py, H):
    pt     = np.array([[[px, py]]], dtype=np.float32)
    result = cv.perspectiveTransform(pt, H)
    return float(result[0][0][0]), float(result[0][0][1])
```

### 1.4 Deliverable ✅
- Calibration overlay: yard lines labelled with yard numbers (tab20 colours)
- Bird's-eye minimap (200×106 px, dynamic `field_length_visible = (n_yl-1)*5+10`)
- Debug rho/theta heatmap with `yl_fit` (dashed white) and `yl_fit2` (solid yellow)

---

## Phase 2 — Single-Player Detection & Tracking

**Goal:** Let the user click on one player; detect and track that player across frames using
YOLO + ByteTrack. Display their bounding box, foot point, and field position on the overlay.

### 2.1 Detection Model

Use **YOLOv8** (pre-trained on COCO, class 0 = person):

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
model = YOLO('yolov8m.pt')   # medium — good speed/accuracy balance
```

Run with ByteTrack to get persistent IDs every frame:

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])
# results[0].boxes.xyxy  → (N,4) float32  [x1,y1,x2,y2]
# results[0].boxes.id    → (N,)  int      [id1, id2, ...]  (None if no track yet)
```

**Why not train from scratch?** COCO pre-training detects people reliably. Fine-tuning on
NFL footage is optional — useful for helmets/partial occlusion but not required for Phase 2.

### 2.2 User Player Selection

The user pauses the video and clicks on a player. The click pixel is matched to the nearest
bounding box to lock a tracker ID.

```python
selected_id   = None   # None = no player selected
click_pending = None   # (x, y) set by mouse callback, consumed each frame

def on_mouse(event, x, y, flags, param):
    global click_pending
    if event == cv.EVENT_LBUTTONDOWN:
        click_pending = (x, y)

cv.setMouseCallback('frame', on_mouse)
```

Each frame, after running YOLO tracking:

```python
if click_pending is not None and boxes is not None:
    cx, cy = click_pending
    click_pending = None
    best_id, best_dist = None, float('inf')
    for (x1, y1, x2, y2), tid in zip(boxes, ids):
        # Check if click is inside box; fall back to nearest box centre
        foot_x = (x1 + x2) / 2
        foot_y = y2
        dist = ((cx - foot_x)**2 + (cy - foot_y)**2) ** 0.5
        if dist < best_dist:
            best_dist, best_id = dist, int(tid)
    selected_id = best_id   # lock this tracker ID
```

Press **`c`** to clear the selection (`selected_id = None`).

### 2.3 ID Re-identification on Track Loss

ByteTrack occasionally drops and re-assigns IDs (occlusion, frame cut). To handle this:

- If `selected_id` is not present in the current frame's IDs, display a **"Player lost"**
  badge but keep `selected_id` set — ByteTrack will often recover the same ID within a few frames.
- If the ID is absent for more than `MAX_LOST_FRAMES = 30` consecutive frames, auto-clear
  `selected_id` and prompt the user to re-select.

```python
lost_counter = 0   # frames since selected_id was last seen

if selected_id not in current_ids:
    lost_counter += 1
    if lost_counter > MAX_LOST_FRAMES:
        selected_id = None
        lost_counter = 0
else:
    lost_counter = 0
```

### 2.4 Foot-Point Extraction

Use the **bottom-center** of the bounding box as the ground-plane contact point:

```python
foot_x = (x1 + x2) / 2
foot_y = y2
field_pos = pixel_to_field(foot_x, foot_y, H)   # reuse Phase 1 transformer
```

### 2.5 Overlay

Draw only the selected player's bounding box and annotations:

```python
if selected_id is not None and selected_id in frame_detections:
    x1, y1, x2, y2 = frame_detections[selected_id]['box']
    fx, fy = frame_detections[selected_id]['field']
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 0), -1)
    cv.putText(frame, f"ID {selected_id}  ({fx:.1f}yd, {fy:.1f}yd)",
               (x1, y1 - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
```

All other detected players: draw a small **grey dot** at their foot point (no box) so the
user can see who to click on next.

```python
for (x1,y1,x2,y2), tid in zip(boxes, ids):
    if int(tid) != selected_id:
        foot_x = int((x1+x2)/2); foot_y = int(y2)
        cv.circle(frame, (foot_x, foot_y), 4, (180,180,180), -1)
```

### 2.6 Minimap Integration

Plot the selected player as a **bright green dot** on the existing bird's-eye minimap:

```python
if H_valid and selected_id in frame_detections:
    fx, fy = frame_detections[selected_id]['field']
    # map field coords → minimap pixel
    px = x0 + int((fx / field_length_visible) * MINIMAP_W)
    py = y0 + int((fy / FIELD_WIDTH) * MINIMAP_H)
    cv.circle(frame, (px, py), 5, (0, 255, 0), -1)
```

### 2.7 Playback Key Bindings (additions)

| Key | Action |
|-----|--------|
| Click | Select player under cursor |
| `c` | Clear player selection |
| (existing) Space, e, q, d | unchanged |

### 2.8 Deliverable

Per frame, with a player selected:
```python
{
  'id':        int,
  'box':       (x1, y1, x2, y2),   # pixel bounding box
  'foot':      (foot_x, foot_y),    # pixel foot point
  'field':     (fx, fy),            # field coords in yards
  'lost':      bool                 # True if ID absent this frame
}
```

---

## Phase 3 — Speed Calculation

**Goal:** Convert frame-to-frame field coordinate deltas into mph / yards per second.

### 3.1 Position History Buffer

```python
from collections import deque

history = deque(maxlen=10)   # last 10 field positions for selected player

def update_history(field_coord):
    history.append(field_coord)
```

Reset `history.clear()` whenever `selected_id` changes.

### 3.2 Speed Formula

```python
video_fps    = cap.get(cv.CAP_PROP_FPS)   # e.g. 29.97
window_frames = 5                          # smooth over 5 frames

if len(history) >= window_frames:
    x0, y0 = history[-window_frames]
    x1, y1 = history[-1]
    distance_yards = ((x1-x0)**2 + (y1-y0)**2) ** 0.5
    time_seconds   = window_frames / video_fps
    speed_yps = distance_yards / time_seconds
    speed_mph = speed_yps * 2.04545        # 1 yd/s = 2.04545 mph
```

### 3.3 Smoothing & Noise Reduction

- **Median filter** over the last 5 speed samples
- **Max plausible speed cap** — discard readings above 25 mph (tracking error)
- **Zero-speed floor** — if distance delta < 0.1 yards, report 0.0 mph

### 3.4 Speed Overlay

Draw current speed on frame above the bounding box:

```python
cv.putText(frame, f"{speed_mph:.1f} mph", (x1, y1 - 28),
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

### 3.5 Deliverable

```python
{'id': 3, 'speed_mph': 14.2, 'field_pos': (23.1, 18.7)}
```

---

## Phase 4 — Integration Layer (Python Backend)

**Goal:** Combine Phases 1–3 into a single processing pipeline that outputs structured data for the frontend.

### 4.1 Pipeline Class

```python
class FieldTracker:
    def __init__(self, video_path):
        self.cap         = cv.VideoCapture(video_path)
        self.yolo        = YOLO('yolov8m.pt')
        self.homography  = None
        self.history     = deque(maxlen=10)
        self.selected_id = None

    def process_frame(self, frame):
        H       = self.calibrate(frame)          # Phase 1
        players = self.detect(frame, H)          # Phase 2
        speed   = self.calc_speed(players)       # Phase 3
        return speed

    def calibrate(self, frame): ...
    def detect(self, frame, H): ...
    def calc_speed(self, players): ...
```

### 4.2 Output Format (JSON per frame)

```json
{
  "frame": 432,
  "timestamp_s": 14.4,
  "selected_player": {
    "id": 3,
    "x": 23.1,
    "y": 18.7,
    "speed_mph": 14.2,
    "lost": false
  },
  "all_foot_points": [[120, 540], [340, 510]]
}
```

### 4.3 API Server (FastAPI)

```bash
pip install fastapi uvicorn python-multipart
```

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/upload` | Accept video file, return `job_id` |
| `GET`  | `/status/{job_id}` | Processing progress (0–100%) |
| `GET`  | `/stream/{job_id}` | SSE stream of frame JSON |
| `POST` | `/select/{job_id}` | Set selected player by click coord `{x, y}` |
| `POST` | `/deselect/{job_id}` | Clear selection |

---

## Phase 5 — Web Frontend

**Goal:** A browser UI that plays the video with overlays and displays the selected player's speed.

### 5.1 Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Framework | React + Vite | Fast dev server, component model |
| Styling | Tailwind CSS | Rapid layout, dark mode |
| Video overlay | HTML5 Canvas | Draw bounding boxes & speed labels |
| Real-time data | EventSource (SSE) | Streams frame data from FastAPI |
| Charts | Recharts | Speed-over-time line chart |

### 5.2 Core UI Components

**`<VideoPlayer>`**
- `<video>` element + synchronized `<canvas>` overlay
- Click on canvas → POST `/select/{jobId}` with `{x, y}` (scaled to video resolution)
- Each frame: draw selected player's bounding box + speed; draw grey dots for all others

**`<SpeedPanel>`**
- Current speed in large text (mph)
- Speed-over-time line chart (last 10 seconds) via Recharts
- "Player lost" warning badge when tracker drops the ID

**`<FieldMinimap>`**
- SVG top-down field; selected player shown as green dot
- Yard line positions drawn from calibration data

### 5.3 SSE Integration

```javascript
const evtSource = new EventSource(`/stream/${jobId}`);
evtSource.onmessage = (e) => {
  const frame = JSON.parse(e.data);
  updateOverlay(frame.selected_player, frame.all_foot_points);
  updateSpeedPanel(frame.selected_player);
  updateMinimap(frame.selected_player);
};
```

---

## Phase 6 — Deployment

**Goal:** Host the app so it's accessible from any browser.

### 6.1 Containerize with Docker

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
RUN apt-get install -y libgl1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine
COPY . .
RUN npm install && npm run build
CMD ["npx", "serve", "dist", "-l", "3000"]
```

### 6.2 Docker Compose

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./uploads:/uploads
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]
```

### 6.3 Cloud Deployment Options

| Option | Best for | Notes |
|--------|----------|-------|
| **Railway** | Quickest launch | Free tier, auto-deploys from GitHub |
| **Render** | Simple VPS-style | Good for GPU-less inference |
| **AWS EC2 + GPU** | Production scale | `g4dn.xlarge` for real-time YOLO |
| **Modal.com** | Serverless GPU | Pay per second, great for batch jobs |

**For GPU inference** (real-time YOLO at 30fps): minimum NVIDIA T4 (`g4dn.xlarge`, ~$0.53/hr)

---

## Implementation Order & Milestones

```
Week 1   Phase 1    COMPLETE — homography + field calibration on imageTestHL.py
Week 2   Phase 2    YOLO single-player selection + ByteTrack ID persistence
Week 3   Phase 3    Speed numbers validated against known play footage
Week 4   Phase 4    FastAPI backend with /upload + /stream + /select endpoints
Week 5   Phase 5    React frontend with video overlay + speed panel
Week 6   Phase 6    Dockerized and deployed to Railway or Render
```

---

## Key Libraries Summary

```
opencv-python       # Phase 1 — keep, extend
numpy               # Phase 1 — keep
matplotlib          # Phase 1 — heatmap debug (tab20, Agg backend)
ultralytics         # Phase 2 — YOLOv8 detection + ByteTrack tracking
fastapi             # Phase 4 — REST + SSE backend
uvicorn             # Phase 4 — ASGI server
python-multipart    # Phase 4 — file uploads
react + vite        # Phase 5 — frontend framework
tailwindcss         # Phase 5 — styling
recharts            # Phase 5 — speed chart
```

---

## Open Questions to Resolve Early

1. **Camera cuts** — broadcast cuts mid-play reset homography. Detect cuts automatically
   (frame-difference threshold > X%) or pause tracking and prompt re-select?
2. **Track-loss UX** — if ByteTrack drops the selected ID, auto-recover or force user re-click?
   `MAX_LOST_FRAMES=30` is the current plan; tune after testing.
3. **Video source** — single broadcast feed only, or All-22 film support?
4. **Latency target** — real-time live stream (needs GPU) vs post-game uploaded clip (CPU ok)?
5. **Hosting budget** — GPU inference costs ~$0.50–$1/hr. Fine for demos; plan for public traffic.
