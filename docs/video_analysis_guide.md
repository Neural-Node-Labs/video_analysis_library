# Video Analysis Library - Complete Usage Guide

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Component Details](#component-details)
4. [Configuration Options](#configuration-options)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## 🚀 Installation

### Requirements

```bash
# requirements.txt
opencv-python==4.8.1.78
numpy==1.24.3
pytest==7.4.3
pytest-cov==4.1.0
pillow==10.1.0

# Optional for production ML models
torch==2.1.0
torchvision==0.16.0
ultralytics==8.0.200  # For YOLO
```

### Setup

```bash
# Clone or download the library
git clone <repository-url>
cd video-analysis-library

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python video_analysis_tests.py
```

---

## ⚡ Quick Start

### Basic Example

```python
from video_analysis import VideoAnalysisPipeline

# Initialize pipeline
pipeline = VideoAnalysisPipeline()

# Process a video file
result = pipeline.process_video(
    video_path="my_video.mp4",
    report_format="markdown"
)

# Check results
if result['status'] == 'success':
    print(f"✅ Analysis complete!")
    print(f"Report: {result['report_path']}")
    print(f"Summary: {result['llm_analysis']['summary']}")
else:
    print(f"❌ Error: {result['error']['message']}")
```

### Output

```
📹 Loading video...
✓ Loaded: 30.5s, 30.0 fps, 915 frames

🎞️  Extracting frames...
✓ Extracted 31 frames

🔧 Preprocessing frames...
✓ Preprocessed 31 frames

🔍 Detecting objects...
✓ Detected 15 objects
  Classes: person, car

🏃 Analyzing motion...
✓ Analyzed 8 movements

🤖 Generating AI summary...
✓ Generated summary with 5 scenes

📄 Generating report...
✓ Report saved to: video_analysis_report_20260118_143022.markdown
```

---

## 🔧 Component Details

### 1. VideoLoaderComponent

**Purpose**: Load and validate video files

```python
from video_analysis import VideoLoaderComponent

loader = VideoLoaderComponent()
result = loader.load(
    video_path="video.mp4",
    validation_mode="strict"  # or "lenient"
)

# Access metadata
metadata = result['metadata']
print(f"Duration: {metadata['duration_sec']}s")
print(f"FPS: {metadata['fps']}")
print(f"Resolution: {metadata['resolution']}")
```

**Supported Formats**: MP4, AVI, MOV, MKV, WEBM

---

### 2. FrameExtractorComponent

**Purpose**: Extract frames at specified intervals

```python
from video_analysis import FrameExtractorComponent

extractor = FrameExtractorComponent()
result = extractor.extract(
    video_handle=video_handle,
    extraction_mode="interval",  # "interval", "keyframe", "all"
    frame_interval_sec=1.0       # Extract 1 frame/sec
)

frames = result['frames']
timestamps = result['timestamps']
```

**Extraction Modes**:
- `interval`: Extract frames at regular time intervals
- `keyframe`: Extract only keyframes (scene changes)
- `all`: Extract every frame (memory intensive)

---

### 3. FramePreprocessorComponent

**Purpose**: Normalize and prepare frames for AI

```python
from video_analysis import FramePreprocessorComponent

preprocessor = FramePreprocessorComponent()
result = preprocessor.preprocess(
    frames=frames,
    target_size=(640, 640),      # Resize to 640x640
    normalization="standard",     # "standard", "minmax", "none"
    color_space="RGB"            # "RGB", "BGR", "GRAYSCALE"
)
```

**Normalization Methods**:
- `standard`: (x - mean) / std
- `minmax`: (x - min) / (max - min)
- `none`: No normalization

---

### 4. ObjectDetectionComponent

**Purpose**: Detect objects using AI models

```python
from video_analysis import ObjectDetectionComponent

detector = ObjectDetectionComponent()
result = detector.detect(
    frames=processed_frames,
    model_type="yolo",           # "yolo", "faster-rcnn", "ssd"
    confidence_threshold=0.5,    # 0.0 to 1.0
    device="cpu"                 # "cpu", "cuda", "mps"
)

# Access detections
for detection in result['detections']:
    print(f"Frame {detection['frame_index']}:")
    for obj in detection['objects']:
        print(f"  - {obj['class']}: {obj['confidence']:.2f}")
```

**Supported Models**:
- YOLO (You Only Look Once) - Fast, real-time
- Faster R-CNN - High accuracy
- SSD (Single Shot Detector) - Balanced

---

### 5. MotionAnalysisComponent

**Purpose**: Track object movement across frames

```python
from video_analysis import MotionAnalysisComponent

analyzer = MotionAnalysisComponent()
result = analyzer.analyze(
    frames=frames,
    detections=detections,
    motion_algorithm="optical_flow"  # "optical_flow", "frame_diff", "object_tracking"
)

# Access motion events
for event in result['motion_events']:
    print(f"Object {event['object_id']}:")
    print(f"  Velocity: {event['velocity']:.2f} px/frame")
    print(f"  Direction: {event['direction']:.2f} radians")
```

---

### 6. LLMAnalysisComponent

**Purpose**: Generate natural language insights

```python
from video_analysis import LLMAnalysisComponent

llm = LLMAnalysisComponent()
result = llm.analyze(
    detections=detections,
    motion_events=motion_events,
    video_metadata=metadata,
    analysis_type="detailed",  # "summary", "detailed", "narrative"
    llm_config={
        "model": "claude-sonnet-4",
        "temperature": 0.7,
        "max_tokens": 2000
    }
)

print(result['summary'])
```

**Analysis Types**:
- `summary`: Brief overview (1-2 paragraphs)
- `detailed`: Comprehensive analysis with scenes
- `narrative`: Story-like description

---

### 7. ReportGeneratorComponent

**Purpose**: Create formatted reports

```python
from video_analysis import ReportGeneratorComponent

generator = ReportGeneratorComponent()
result = generator.generate(
    llm_analysis=llm_result,
    detections=detections,
    motion_events=motion_events,
    report_format="markdown",  # "json", "markdown", "html", "pdf"
    include_visualizations=True
)

print(f"Report saved to: {result['report_path']}")
```

---

## ⚙️ Configuration Options

### Complete Pipeline Configuration

```python
pipeline = VideoAnalysisPipeline()

result = pipeline.process_video(
    video_path="video.mp4",
    
    # Frame extraction settings
    extraction_config={
        "extraction_mode": "interval",
        "frame_interval_sec": 0.5,  # Extract every 0.5 seconds
        "fps_target": None
    },
    
    # Object detection settings
    detection_config={
        "model_type": "yolo",
        "confidence_threshold": 0.6,
        "device": "cuda",  # Use GPU if available
    },
    
    # Report settings
    report_format="json"  # Output format
)
```

---

## 🎯 Advanced Usage

### Custom Component Integration

```python
from video_analysis import (
    VideoLoaderComponent,
    FrameExtractorComponent,
    FramePreprocessorComponent
)

# Manual component orchestration
loader = VideoLoaderComponent()
extractor = FrameExtractorComponent()
preprocessor = FramePreprocessorComponent()

# Step-by-step processing
video_result = loader.load("video.mp4")
frame_result = extractor.extract(
    video_result['video_handle'],
    extraction_mode="keyframe"
)
processed = preprocessor.preprocess(
    frame_result['frames'],
    target_size=(416, 416)  # YOLO-optimized size
)
```

### Batch Processing

```python
import glob
from video_analysis import VideoAnalysisPipeline

pipeline = VideoAnalysisPipeline()

# Process all videos in a directory
video_files = glob.glob("videos/*.mp4")

for video_path in video_files:
    print(f"Processing: {video_path}")
    result = pipeline.process_video(
        video_path=video_path,
        report_format="json"
    )
    
    if result['status'] == 'success':
        print(f"✅ Completed: {result['report_path']}")
    else:
        print(f"❌ Failed: {result['error']['message']}")
```

### Real-Time Processing

```python
from video_analysis import FrameExtractorComponent, ObjectDetectionComponent

# For webcam or RTSP stream
import cv2

cap = cv2.VideoCapture(0)  # 0 for webcam
detector = ObjectDetectionComponent()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects in real-time
    result = detector.detect(
        frames=[frame],
        confidence_threshold=0.7
    )
    
    # Display results
    for obj in result['detections'][0]['objects']:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, obj['class'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📚 API Reference

### VideoAnalysisPipeline

```python
class VideoAnalysisPipeline:
    def process_video(
        video_path: str,
        extraction_config: Dict = None,
        detection_config: Dict = None,
        report_format: str = "json"
    ) -> Dict
```

**Returns**:
```python
{
    "status": "success" | "error",
    "metadata": {...},           # Video metadata
    "detection_summary": {...},  # Detection statistics
    "motion_summary": {...},     # Motion statistics
    "llm_analysis": {...},       # AI-generated insights
    "report_path": "string",     # Path to report file
    "trace_id": "uuid"          # For debugging
}
```

---

## 🚀 Performance Optimization

### 1. GPU Acceleration

```python
# Use CUDA for faster inference
result = pipeline.process_video(
    video_path="video.mp4",
    detection_config={
        "device": "cuda",  # Requires NVIDIA GPU
        "model_type": "yolo"
    }
)
```

### 2. Frame Sampling

```python
# Process fewer frames for faster results
extraction_config = {
    "extraction_mode": "interval",
    "frame_interval_sec": 2.0  # Sample every 2 seconds
}
```

### 3. Caching

```python
from video_analysis import CacheManagerComponent

# Results are automatically cached
cache = CacheManagerComponent()

# Manual cache management
cache_key = f"video_{video_path}_detections"
cached = cache.get(cache_key)

if cached['cache_hit']:
    detections = cached['data']
else:
    # Process and cache
    detections = detector.detect(frames)
    cache.set(cache_key, detections)
```

### 4. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_frame_batch(batch):
    return detector.detect(batch)

# Process frames in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    # Split frames into batches
    batch_size = 10
    batches = [frames[i:i+batch_size] 
               for i in range(0, len(frames), batch_size)]
    
    results = list(executor.map(process_frame_batch, batches))
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "Video file not found"
```python
# Ensure path is correct
import os
assert os.path.exists("video.mp4"), "File not found"

# Use absolute paths
from pathlib import Path
video_path = Path("video.mp4").resolve()
```

#### 2. "Memory Error"
```python
# Reduce frame extraction rate
extraction_config = {
    "frame_interval_sec": 5.0  # Extract less frequently
}

# Or use keyframe mode
extraction_config = {
    "extraction_mode": "keyframe"
}
```

#### 3. "Model inference slow"
```python
# Use GPU
detection_config = {"device": "cuda"}

# Or reduce frame size
preprocessor.preprocess(frames, target_size=(320, 320))
```

#### 4. Check Logs
```python
# System errors
with open('system.log', 'r') as f:
    print(f.read())

# LLM interactions
with open('llm_interaction.log', 'r') as f:
    import json
    for line in f:
        print(json.loads(line))
```

---

## 🔒 Security Best Practices

### 1. Input Validation

```python
from pathlib import Path

def safe_load_video(video_path):
    # Prevent path traversal
    path = Path(video_path).resolve()
    
    # Ensure file is in allowed directory
    allowed_dir = Path("/safe/videos").resolve()
    if not str(path).startswith(str(allowed_dir)):
        raise ValueError("Invalid video path")
    
    return path
```

### 2. API Key Protection

```python
import os

# Never hardcode API keys
llm_config = {
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "model": "claude-sonnet-4"
}
```

### 3. Resource Limits

```python
# Limit frame extraction
MAX_FRAMES = 1000

result = extractor.extract(video_handle)
if result['frame_count'] > MAX_FRAMES:
    frames = result['frames'][:MAX_FRAMES]
```

### 4. Error Isolation

```python
try:
    result = pipeline.process_video("video.mp4")
except Exception as e:
    # Log error without exposing internals
    logger.log("system", "ERROR", "Processing failed", 
              {"error": str(e)[:100]})  # Truncate sensitive info
```

---

## 📊 Example Reports

### JSON Report Structure

```json
{
  "summary": "30-second video showing a person walking...",
  "scene_descriptions": [
    {
      "time_range": [0.0, 10.0],
      "description": "Person enters from left side",
      "objects_present": ["person", "car"],
      "actions": ["walking", "entering"]
    }
  ],
  "key_moments": [
    {
      "timestamp": 5.2,
      "description": "Person stops to check phone",
      "importance_score": 0.85
    }
  ],
  "generated_at": "2026-01-18T14:30:22"
}
```

### Markdown Report

```markdown
# Video Analysis Report

**Generated**: 2026-01-18T14:30:22

## Summary

30-second video showing a person walking across a parking lot...

## Scene Descriptions

### Scene 0.0s - 10.0s

**Description**: Person enters from left side

**Objects Present**: person, car

**Actions**: walking, entering

## Key Moments

- **5.2s**: Person stops to check phone (Importance: 0.85)
```

---

## 🎓 Learning Resources

- **Component-Based Development**: [CBD Methodology](https://en.wikipedia.org/wiki/Component-based_software_engineering)
- **Computer Vision**: [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- **Object Detection**: [YOLO Documentation](https://docs.ultralytics.com/)
- **Video Processing**: [FFmpeg Guide](https://ffmpeg.org/documentation.html)

---

## 📞 Support

For issues, questions, or contributions:
- Check logs: `system.log` and `llm_interaction.log`
- Review test suite: `python video_analysis_tests.py`
- Trace ID for debugging: Available in all error responses

---

**Version**: 1.0.0  
**Last Updated**: January 18, 2026  
**License**: MIT
