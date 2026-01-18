# 🚀 Quick Start Guide - Video Analysis Library v2.0

Get started in 5 minutes!

---

## Step 1: Install (2 minutes)

```bash
# Clone repository
git clone https://github.com/your-repo/video-analysis-library.git
cd video-analysis-library

# Install dependencies
pip install opencv-python numpy requests pillow

# Verify installation
python -c "import cv2; print('✅ Ready to go!')"
```

---

## Step 2: Get API Key (1 minute)

Choose your preferred LLM provider:

### Option A: DeepSeek (Recommended - Most Cost-Effective)
1. Visit https://platform.deepseek.com/
2. Sign up and get API key
3. Set environment variable:
```bash
export DEEPSEEK_API_KEY="your-key-here"
```

### Option B: Claude (Best Quality)
1. Visit https://console.anthropic.com/
2. Get API key
3. Set variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Option C: OpenAI
```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Step 3: Run Your First Analysis (2 minutes)

### Example 1: Basic Video Analysis

Create `my_first_analysis.py`:

```python
from video_analysis_v2 import VideoAnalysisPipelineV2
import os

# Initialize
pipeline = VideoAnalysisPipelineV2()

# Analyze video
result = pipeline.process_video(
    video_path="your_video.mp4",
    llm_config={
        "provider": "deepseek",  # or "claude" or "openai"
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    },
    report_format="markdown"
)

# Check results
if result['status'] == 'success':
    print("✅ Success!")
    print(f"📄 Report: {result['report_path']}")
    print(f"\n📝 Summary:\n{result['llm_analysis']['summary']}")
else:
    print(f"❌ Error: {result['error']['message']}")
```

Run it:
```bash
python my_first_analysis.py
```

**Expected Output:**
```
🎬 VIDEO ANALYSIS PIPELINE v2.0
================================================================

📹 Loading video...
✓ Loaded: 30.5s, 30.0 fps, 915 frames

🎞️  Extracting frames...
✓ Extracted 31 frames

🔍 Detecting objects...
✓ Detected 15 objects

🤖 Generating AI summary...
✓ Analysis complete using deepseek

📄 Generating report.md...
✓ Report saved to: report.md

✅ Success!
📄 Report: report.md

📝 Summary:
This video shows...
```

---

## Step 4: View Your Report

Open `report.md` - it contains:
- ✅ Executive summary
- ✅ Detailed timeline
- ✅ Object detection stats
- ✅ Motion analysis
- ✅ Key moments

---

## 🎯 Next Steps

### Find Specific Objects in Video

```python
# Create find_object.py
from video_analysis_v2 import VideoAnalysisPipelineV2

pipeline = VideoAnalysisPipelineV2()

result = pipeline.process_video(
    video_path="security_footage.mp4",
    reference_image_path="suspect_photo.jpg",  # Object to find
    enable_visual_search=True,
    llm_config={"provider": "claude"}  # Claude best for vision
)

# See where object was found
for match in result['visual_search_results']:
    print(f"Found at {match['timestamp']:.2f}s (confidence: {match['confidence']:.2f})")
```

### Process Multiple Videos

```python
# Create batch_process.py
import glob
from video_analysis_v2 import VideoAnalysisPipelineV2

pipeline = VideoAnalysisPipelineV2()

for video in glob.glob("videos/*.mp4"):
    print(f"Processing: {video}")
    result = pipeline.process_video(
        video_path=video,
        llm_config={"provider": "deepseek"}
    )
    print(f"✅ Done: {result['report_path']}\n")
```

### Save Money with Smart Configuration

```python
# Optimize costs
result = pipeline.process_video(
    video_path="long_video.mp4",
    extraction_config={
        "frame_interval_sec": 5.0  # Sample less frequently
    },
    llm_config={
        "provider": "deepseek",      # Cheapest option
        "temperature": 0.5,          # Lower = fewer tokens
        "fallback_provider": "claude"  # Backup if needed
    }
)

print(f"Tokens used: {result['llm_analysis']['tokens_used']}")
print(f"Provider: {result['llm_analysis']['provider_used']}")
```

---

## 🆘 Troubleshooting

### "API key not found"
```bash
# Check environment variable
echo $DEEPSEEK_API_KEY

# If empty, set it:
export DEEPSEEK_API_KEY="your-key"
```

### "Video file not found"
```python
# Use absolute path
from pathlib import Path
video_path = Path("videos/sample.mp4").resolve()
print(video_path)  # Verify it exists
```

### "Out of memory"
```python
# Extract fewer frames
extraction_config = {
    "frame_interval_sec": 5.0  # Every 5 seconds instead of 1
}
```

### Check Logs
```bash
# System errors
cat system.log

# LLM interactions
cat llm_interaction.log
```

---

## 📊 Cost Comparison

Processing a 5-minute video:

| Provider | Cost per Analysis | Quality |
|----------|------------------|---------|
| DeepSeek | ~$0.02 | Good |
| Claude | ~$0.15 | Excellent |
| OpenAI | ~$0.10 | Very Good |

**Recommendation:** Start with DeepSeek for development, upgrade to Claude for production if quality matters.

---

## 🎓 Learn More

- **Full Documentation**: [README.md](README.md)
- **API Reference**: See README.md § API Reference
- **Examples**: Check `examples/` directory
- **Tests**: Run `pytest video_analysis_v2_tests.py`

---

## ✨ Feature Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Object Detection | ✅ | ✅ |
| Motion Tracking | ✅ | ✅ |
| LLM Summary | ✅ Claude only | ✅ Multi-provider |
| Visual Search | ❌ | ✅ New! |
| DeepSeek Support | ❌ | ✅ New! |
| Enhanced Reports | Basic | ✅ Advanced |
| Provider Fallback | ❌ | ✅ New! |
| Token Tracking | ❌ | ✅ New! |

---

## 🚀 Ready to Go!

You're now ready to analyze videos with AI! Try these commands:

```bash
# Basic analysis
python my_first_analysis.py

# Find object
python find_object.py

# Batch processing
python batch_process.py

# Run tests
pytest video_analysis_v2_tests.py -v
```

**Happy analyzing! 🎬**

---

*For questions: Check [README.md](README.md) or system logs*
