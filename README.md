# 🎬 Video Analysis Library v2.0

AI-powered video analysis library with frame-by-frame object detection, motion tracking, and intelligent summarization using LLMs.

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/your-repo)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

---

## ✨ Features

### Core Capabilities
- 🎞️ **Frame Extraction** - Extract frames at intervals, keyframes, or all frames
- 🔍 **Object Detection** - AI-powered object detection (YOLO, Faster R-CNN, SSD)
- 🏃 **Motion Analysis** - Track objects and analyze movement patterns
- 🤖 **LLM Analysis** - Natural language scene understanding and summarization
- 📄 **Report Generation** - Comprehensive reports in JSON, Markdown, HTML formats

### 🆕 Version 2.0 Enhancements
- ✅ **DeepSeek Integration** - Cost-effective LLM provider
- ✅ **Multi-Provider LLM** - Support for Claude, OpenAI, and DeepSeek
- ✅ **Visual Object Search** - Find specific objects from reference images
- ✅ **Enhanced Reports** - Auto-generated `report.md` with statistics
- ✅ **Provider Fallback** - Automatic fallback to backup LLM provider
- ✅ **Token Tracking** - Monitor and optimize LLM usage

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Configuration](#-configuration)
4. [Usage Examples](#-usage-examples)
5. [API Reference](#-api-reference)
6. [Advanced Features](#-advanced-features)
7. [Troubleshooting](#-troubleshooting)
8. [Contributing](#-contributing)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster processing

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/video-analysis-library.git
cd video-analysis-library
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
opencv-python==4.8.1.78
numpy==1.24.3
requests==2.31.0
pillow==10.1.0
pytest==7.4.3
pytest-cov==4.1.0

# Optional: For GPU acceleration
torch==2.1.0
torchvision==0.16.0

# Optional: For production object detection
ultralytics==8.0.200
```

### Step 3: Set Up API Keys

Create a `.env` file in the project root:

```bash
# Choose your LLM provider(s)

# Anthropic Claude
ANTHROPIC_API_KEY=your_claude_api_key_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# DeepSeek (recommended for cost-effectiveness)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

**Getting API Keys:**
- **Claude**: https://console.anthropic.com/
- **OpenAI**: https://platform.openai.com/api-keys
- **DeepSeek**: https://platform.deepseek.com/

### Step 4: Verify Installation

```bash
python -c "import cv2, numpy; print('✅ Installation successful!')"
```

---

## ⚡ Quick Start

### Basic Usage (30 seconds)

```python
from video_analysis_v2 import VideoAnalysisPipelineV2
import os

# Initialize pipeline
pipeline = VideoAnalysisPipelineV2()

# Process video with DeepSeek (cost-effective)
result = pipeline.process_video(
    video_path="my_video.mp4",
    llm_config={
        "provider": "deepseek",
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    },
    report_format="markdown"
)

# Check results
if result['status'] == 'success':
    print(f"✅ Analysis complete!")
    print(f"📄 Report: {result['report_path']}")
    print(f"📝 Summary: {result['llm_analysis']['summary']}")
else:
    print(f"❌ Error: {result['error']['message']}")
```

**Output:**
```
🎬 VIDEO ANALYSIS PIPELINE v2.0
================================================================

📹 Loading video...
✓ Loaded: 30.5s, 30.0 fps, 915 frames

🎞️  Extracting frames...
✓ Extracted 31 frames

🔧 Preprocessing frames...
✓ Preprocessed 31 frames

🔍 Detecting objects...
✓ Detected 15 objects

🏃 Analyzing motion...
✓ Analyzed 8 movements

🤖 Generating AI summary...
✓ Analysis complete using deepseek

📄 Generating report.md...
✓ Report saved to: report.md

================================================================
✅ ANALYSIS COMPLETE
================================================================
```

---

## ⚙️ Configuration

### LLM Provider Configuration

#### Option 1: DeepSeek (Recommended - Most Cost-Effective)

```python
llm_config = {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 2000
}
```

**Pricing**: ~$0.14 per 1M tokens (input), ~$0.28 per 1M tokens (output)

#### Option 2: Claude (Best Quality)

```python
llm_config = {
    "provider": "claude",
    "model": "claude-sonnet-4-20250514",
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 2000
}
```

#### Option 3: OpenAI GPT

```python
llm_config = {
    "provider": "openai",
    "model": "gpt-4-turbo-preview",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 2000
}
```

#### Multi-Provider with Fallback

```python
llm_config = {
    "provider": "deepseek",
    "fallback_provider": "claude",  # Use if DeepSeek fails
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "temperature": 0.7
}
```

### Frame Extraction Configuration

```python
extraction_config = {
    "extraction_mode": "interval",  # "interval", "keyframe", "all"
    "frame_interval_sec": 1.0,      # Extract 1 frame per second
    "fps_target": None              # Optional: target FPS
}
```

**Modes:**
- `interval`: Extract frames at regular intervals (recommended)
- `keyframe`: Extract only keyframes (scene changes)
- `all`: Extract every frame (memory intensive)

### Object Detection Configuration

```python
detection_config = {
    "model_type": "yolo",           # "yolo", "faster-rcnn", "ssd"
    "confidence_threshold": 0.5,    # 0.0 to 1.0
    "device": "cpu"                 # "cpu", "cuda", "mps"
}
```

### Visual Search Configuration

```python
search_config = {
    "provider": "claude",           # Vision-capable model
    "model": "claude-3-sonnet-20240229",
    "similarity_threshold": 0.75,   # 0.0 to 1.0
    "max_matches": 10,
    "description_prompt": "Describe this object in detail"
}
```

---

## 💡 Usage Examples

### Example 1: Basic Video Analysis

```python
from video_analysis_v2 import VideoAnalysisPipelineV2

pipeline = VideoAnalysisPipelineV2()

result = pipeline.process_video(
    video_path="videos/sample.mp4",
    llm_config={"provider": "deepseek"},
    report_format="markdown"
)

print(f"Report generated: {result['report_path']}")
```

### Example 2: Visual Object Search

Find a specific object (from a reference image) in your video:

```python
pipeline = VideoAnalysisPipelineV2()

# Provide a reference image of the object to find
result = pipeline.process_video(
    video_path="videos/security_footage.mp4",
    reference_image_path="images/suspect_jacket.jpg",
    enable_visual_search=True,
    llm_config={
        "provider": "claude",  # Claude has excellent vision capabilities
        "model": "claude-3-sonnet-20240229"
    }
)

# Access search results
if result['status'] == 'success':
    matches = result['visual_search_results']
    print(f"\n🔎 Found object in {len(matches)} frames:")
    
    for match in matches:
        print(f"  ✓ Frame {match['frame_index']} @ {match['timestamp']:.2f}s")
        print(f"    Confidence: {match['confidence']:.2f}")
        print(f"    Description: {match['match_description']}")
```

### Example 3: Cost-Optimized Processing

```python
# Use DeepSeek for cost savings with Claude fallback for reliability
pipeline = VideoAnalysisPipelineV2()

result = pipeline.process_video(
    video_path="videos/long_video.mp4",
    extraction_config={
        "extraction_mode": "interval",
        "frame_interval_sec": 2.0  # Extract every 2 seconds (fewer frames)
    },
    llm_config={
        "provider": "deepseek",
        "fallback_provider": "claude",
        "temperature": 0.5  # Lower temperature = more focused, uses fewer tokens
    }
)

print(f"Tokens used: {result['llm_analysis']['tokens_used']}")
print(f"Provider used: {result['llm_analysis']['provider_used']}")
```

### Example 4: Batch Processing

Process multiple videos:

```python
import glob
from video_analysis_v2 import VideoAnalysisPipelineV2

pipeline = VideoAnalysisPipelineV2()

# Get all videos in directory
video_files = glob.glob("videos/*.mp4")

for video_path in video_files:
    print(f"\n📹 Processing: {video_path}")
    
    result = pipeline.process_video(
        video_path=video_path,
        llm_config={"provider": "deepseek"},
        report_format="markdown"
    )
    
    if result['status'] == 'success':
        print(f"  ✅ Report: {result['report_path']}")
    else:
        print(f"  ❌ Failed: {result['error']['message']}")
```

### Example 5: Custom Analysis with Component-Level Control

```python
from video_analysis_v2 import (
    MultiProviderLLMComponent,
    VisualObjectSearchComponent,
    MarkdownReportGeneratorComponent
)

# Manual component orchestration for fine control
llm = MultiProviderLLMComponent()
visual_search = VisualObjectSearchComponent()
report_gen = MarkdownReportGeneratorComponent()

# Step 1: Visual search
search_result = visual_search.search(
    reference_image_path="target.jpg",
    video_frames=frames,
    frame_timestamps=timestamps,
    search_config={
        "provider": "claude",
        "similarity_threshold": 0.8
    }
)

# Step 2: LLM analysis
llm_result = llm.analyze(
    detections=detections,
    motion_events=motion_events,
    video_metadata=metadata,
    llm_config={
        "provider": "deepseek",
        "fallback_provider": "claude"
    }
)

# Step 3: Generate report
report = report_gen.generate(
    llm_analysis=llm_result,
    detections=detections,
    motion_events=motion_events,
    video_metadata=metadata,
    visual_search_results=search_result['matches']
)

print(f"Custom report: {report['report_path']}")
```

---

## 📚 API Reference

### VideoAnalysisPipelineV2

Main pipeline for complete video analysis.

```python
class VideoAnalysisPipelineV2:
    def process_video(
        video_path: str,
        reference_image_path: str = None,
        extraction_config: Dict = None,
        detection_config: Dict = None,
        llm_config: Dict = None,
        report_format: str = "markdown",
        enable_visual_search: bool = None
    ) -> Dict
```

**Parameters:**
- `video_path` (str): Path to video file
- `reference_image_path` (str, optional): Path to reference image for visual search
- `extraction_config` (dict, optional): Frame extraction settings
- `detection_config` (dict, optional): Object detection settings
- `llm_config` (dict, optional): LLM provider and model settings
- `report_format` (str): Output format - "markdown", "json", "html"
- `enable_visual_search` (bool, optional): Enable visual object search

**Returns:**
```python
{
    "status": "success" | "error",
    "metadata": {
        "duration_sec": float,
        "fps": float,
        "resolution": tuple,
        "total_frames": int
    },
    "llm_analysis": {
        "summary": str,
        "scene_descriptions": list,
        "key_moments": list,
        "provider_used": str,
        "tokens_used": int
    },
    "visual_search_results": list | None,
    "report_path": str,
    "trace_id": str
}
```

### MultiProviderLLMComponent

```python
class MultiProviderLLMComponent:
    def analyze(
        detections: List[Dict],
        motion_events: List[Dict],
        video_metadata: Dict,
        analysis_type: str = "summary",
        llm_config: Dict = None
    ) -> Dict
```

### VisualObjectSearchComponent

```python
class VisualObjectSearchComponent:
    def search(
        reference_image_path: str,
        video_frames: List[np.ndarray],
        frame_timestamps: List[float],
        search_config: Dict = None
    ) -> Dict
```

### MarkdownReportGeneratorComponent

```python
class MarkdownReportGeneratorComponent:
    def generate(
        llm_analysis: Dict,
        detections: List[Dict],
        motion_events: List[Dict],
        video_metadata: Dict,
        visual_search_results: List[Dict] = None,
        include_statistics: bool = True,
        include_timeline: bool = True
    ) -> Dict
```

---

## 🎯 Advanced Features

### 1. GPU Acceleration

Enable GPU for faster object detection:

```python
result = pipeline.process_video(
    video_path="video.mp4",
    detection_config={
        "device": "cuda",  # Requires NVIDIA GPU
        "model_type": "yolo"
    }
)
```

### 2. Caching for Performance

Results are automatically cached:

```python
from video_analysis import CacheManagerComponent

cache = CacheManagerComponent()

# Check cache before processing
cache_key = f"video_{video_path}_analysis"
cached_result = cache.get(cache_key)

if cached_result['cache_hit']:
    result = cached_result['data']
else:
    result = pipeline.process_video(video_path)
    cache.set(cache_key, result)
```

### 3. Real-Time Video Stream Processing

Process live video streams:

```python
import cv2
from video_analysis_v2 import ObjectDetectionComponent

detector = ObjectDetectionComponent()
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect(
        frames=[frame],
        confidence_threshold=0.7
    )
    
    # Draw detections
    for obj in result['detections'][0]['objects']:
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Live Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Custom Prompts for LLM Analysis

```python
llm_config = {
    "provider": "claude",
    "custom_prompt": """Analyze this video with focus on:
    1. Safety violations
    2. Unusual activities
    3. Time-based patterns
    
    Provide actionable insights."""
}
```

---

## 📊 Generated Report Structure

The `report.md` file includes:

### 1. Executive Summary
AI-generated overview of the entire video

### 2. Video Information
- Duration, resolution, FPS
- Codec and technical details

### 3. Analysis Results
- Object detection summary
- Motion analysis summary
- Visual search results (if enabled)

### 4. Detailed Timeline
Scene-by-scene breakdown with:
- Time ranges
- Objects present
- Actions detected
- Key moments

### 5. Visual Search Matches
Frame-by-frame matches with:
- Timestamps
- Confidence scores
- Match descriptions

### 6. Statistics
- Detection distribution table
- Frame analysis coverage
- Motion statistics

### 7. Technical Details
- LLM provider used
- Token usage
- Processing time

**Example Report:**

```markdown
# 🎬 Video Analysis Report

**Generated by**: Video Analysis Library v2.0
**Timestamp**: 2026-01-18 14:30:22

## 📋 Executive Summary

This 30-second video captures a person walking across a parking lot...

## 📹 Video Information

| Property | Value |
|----------|-------|
| Duration | 30.5 seconds |
| Resolution | 1920x1080 |
| FPS | 30.0 |

## 🔍 Analysis Results

### Object Detection Summary
- Total Objects: 42
- Unique Classes: person, car, tree

### Visual Search Results
- Reference Object: Red backpack
- Matches Found: 8 frames
- Best Match: Frame 145 @ 4.83s (confidence: 0.92)

...
```

---

## 🔧 Troubleshooting

### Issue 1: "API key not found"

**Solution:**
```bash
# Create .env file
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# Or export in terminal
export DEEPSEEK_API_KEY="your_key_here"
```

### Issue 2: "Out of memory"

**Solution:**
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

### Issue 3: "Video file not found"

**Solution:**
```python
# Use absolute paths
from pathlib import Path

video_path = Path("videos/sample.mp4").resolve()
assert video_path.exists(), "Video not found"
```

### Issue 4: "Slow processing"

**Solution:**
```python
# Enable GPU
detection_config = {"device": "cuda"}

# Reduce frames
extraction_config = {"frame_interval_sec": 2.0}

# Use faster model
detection_config = {"model_type": "yolo"}  # Faster than Faster-RCNN
```

### Check Logs

```python
# System errors
with open('system.log', 'r') as f:
    print(f.read())

# LLM interactions
with open('llm_interaction.log', 'r') as f:
    for line in f:
        print(json.loads(line))
```

---

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
python video_analysis_tests.py

# Run with coverage
pytest video_analysis_tests.py --cov=video_analysis_v2 --cov-report=html

# Run specific test
pytest video_analysis_tests.py::TestVisualObjectSearchComponent
```

---

## 🔒 Security Best Practices

### 1. API Key Management

```python
# ✅ Good: Use environment variables
import os
api_key = os.getenv("DEEPSEEK_API_KEY")

# ❌ Bad: Hardcode keys
api_key = "sk-1234567890"  # Never do this!
```

### 2. Input Validation

```python
# Validate video paths
from pathlib import Path

def safe_video_path(path):
    p = Path(path).resolve()
    allowed_dir = Path("/safe/videos").resolve()
    
    if not str(p).startswith(str(allowed_dir)):
        raise ValueError("Invalid path")
    
    return p
```

### 3. Resource Limits

```python
# Limit frame extraction
MAX_FRAMES = 1000

extraction_config = {
    "frame_interval_sec": max(1.0, duration / MAX_FRAMES)
}
```

---

## 📖 Examples Directory

Check the `examples/` directory for more:

```
examples/
├── basic_analysis.py
├── visual_search_demo.py
├── batch_processing.py
├── real_time_stream.py
├── custom_llm_prompts.py
└── cost_optimization.py
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/video-analysis-library.git
cd video-analysis-library

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Check code style
flake8 video_analysis_v2.py
black video_analysis_v2.py
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: [Full Docs](https://docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@example.com

---

## 🙏 Acknowledgments

- OpenCV for video processing
- Anthropic Claude for LLM capabilities
- DeepSeek for cost-effective AI
- YOLO for object detection

---

## 📈 Changelog

### v2.0.0 (2026-01-18)
- ✅ Added DeepSeek LLM integration
- ✅ Multi-provider LLM support
- ✅ Visual object search feature
- ✅ Enhanced report.md generation
- ✅ Provider fallback mechanism

### v1.0.0 (2026-01-17)
- Initial release
- Basic video analysis pipeline
- Object detection and motion tracking

---

**Made with ❤️ by the Video Analysis Team**
