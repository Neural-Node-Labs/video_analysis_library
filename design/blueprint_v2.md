# Video Analysis Library v2.0 - Enhancement Blueprint

## Version 2.0 New Features

### 🆕 Enhancement Overview

1. **DeepSeek LLM Integration** - Multi-model support (Claude, GPT, DeepSeek)
2. **Enhanced Report Generation** - Auto-generate comprehensive report.md
3. **Visual Object Search** - Find specific objects from reference images across video frames

---

## New Component: DeepSeekLLMComponent

### Component Identity
**Name**: DeepSeekLLMComponent  
**Purpose**: Integration with DeepSeek AI models for cost-effective LLM analysis

### IN Schema
```json
{
  "detections": "list[FrameDetection]",
  "motion_events": "list[MotionEvent]",
  "video_metadata": "object",
  "analysis_type": "enum[summary, detailed, narrative]",
  "llm_config": {
    "provider": "enum[claude, openai, deepseek]",
    "model": "string",
    "api_key": "string",
    "temperature": "float",
    "max_tokens": "int",
    "base_url": "string|null"
  }
}
```

### OUT Schema
```json
{
  "summary": "string",
  "scene_descriptions": "list[SceneDescription]",
  "key_moments": "list[KeyMoment]",
  "provider_used": "string",
  "tokens_used": "int",
  "status": "enum[success, error]"
}
```

### Error Schema
```json
{
  "error_code": "enum[API_ERROR, RATE_LIMIT, INVALID_PROVIDER, AUTH_FAILED]",
  "message": "string",
  "provider": "string",
  "trace_id": "uuid"
}
```

---

## Enhanced Component: MarkdownReportGeneratorComponent

### Component Identity
**Name**: MarkdownReportGeneratorComponent  
**Purpose**: Generate comprehensive, well-formatted report.md with visualizations

### IN Schema
```json
{
  "llm_analysis": "object",
  "detections": "list[FrameDetection]",
  "motion_events": "list[MotionEvent]",
  "video_metadata": "object",
  "visual_search_results": "list[VisualSearchMatch]|null",
  "include_statistics": "bool",
  "include_timeline": "bool",
  "include_visualizations": "bool"
}
```

### OUT Schema
```json
{
  "report_path": "string",
  "report_content": "string",
  "metadata": {
    "generated_at": "datetime",
    "sections": "list[string]",
    "size_bytes": "int",
    "has_visualizations": "bool"
  },
  "status": "enum[success, error]"
}
```

---

## New Component: VisualObjectSearchComponent

### Component Identity
**Name**: VisualObjectSearchComponent  
**Purpose**: Find specific objects from reference images across video frames using LLM vision

### IN Schema
```json
{
  "reference_image_path": "string",
  "video_frames": "list[numpy.ndarray]",
  "frame_timestamps": "list[float]",
  "search_config": {
    "provider": "enum[claude, openai, deepseek]",
    "model": "string",
    "similarity_threshold": "float",
    "max_matches": "int",
    "description_prompt": "string|null"
  }
}
```

### OUT Schema
```json
{
  "matches": "list[VisualSearchMatch]",
  "search_summary": {
    "total_matches": "int",
    "best_match_frame": "int|null",
    "best_match_confidence": "float|null",
    "frames_analyzed": "int"
  },
  "reference_description": "string",
  "status": "enum[success, error]"
}

VisualSearchMatch = {
  "frame_index": "int",
  "timestamp": "float",
  "confidence": "float",
  "match_description": "string",
  "bounding_box": "tuple[int, int, int, int]|null",
  "similarity_score": "float"
}
```

### Error Schema
```json
{
  "error_code": "enum[IMAGE_LOAD_FAILED, VISION_API_ERROR, NO_MATCHES_FOUND]",
  "message": "string",
  "trace_id": "uuid"
}
```

### Trace Points
- reference_image_loaded
- vision_analysis_start
- frame_comparison_batch (every 10 frames)
- matches_found
- visual_search_complete

---

## Updated Component: LLMAnalysisComponent v2.0

### Enhanced Features
- Multi-provider support (Claude, OpenAI, DeepSeek)
- Provider fallback mechanism
- Token usage tracking
- Cost optimization

### Updated IN Schema
```json
{
  "detections": "list[FrameDetection]",
  "motion_events": "list[MotionEvent]",
  "video_metadata": "object",
  "analysis_type": "enum[summary, detailed, narrative]",
  "llm_config": {
    "provider": "enum[claude, openai, deepseek]",
    "model": "string",
    "api_key": "string|null",
    "temperature": "float",
    "max_tokens": "int",
    "base_url": "string|null",
    "fallback_provider": "string|null"
  }
}
```

---

## Enhanced Pipeline Flow v2.0

```
VideoFile → VideoLoader → FrameExtractor → FramePreprocessor
                                                  ↓
                                          ObjectDetection
                                                  ↓
[Optional: Reference Image] → VisualObjectSearch ┐
                                                  ↓
                                          MotionAnalysis
                                                  ↓
                                    LLMAnalysis (Multi-Provider)
                                                  ↓
                                  MarkdownReportGenerator → report.md
                                                  
[CacheManager + LoggingComponent support all stages]
```

---

## New Adaptors

### VisualSearchAdaptor
Transforms: ReferenceImage + VideoFrames → VisualSearchResults

### MultiProviderLLMAdaptor
Transforms: LLMRequest → Provider-specific API calls with fallback

### EnhancedReportAdaptor
Transforms: AllAnalysisResults → Comprehensive Markdown Report

---

## report.md Structure

```markdown
# Video Analysis Report

## Executive Summary
[AI-generated overview]

## Video Information
- **File**: video.mp4
- **Duration**: 30.5 seconds
- **Resolution**: 1920x1080
- **FPS**: 30.0
- **Total Frames**: 915

## Analysis Results

### Object Detection Summary
- **Total Objects Detected**: 42
- **Unique Classes**: person (15), car (8), dog (2)
- **Average Confidence**: 0.87

### Motion Analysis Summary
- **Movement Events**: 12
- **Average Velocity**: 45.3 px/frame
- **Active Time**: 85% of video

### Visual Search Results (if applicable)
- **Reference Object**: [description]
- **Matches Found**: 8 frames
- **Best Match**: Frame 145 @ 4.83s (confidence: 0.92)

## Detailed Timeline

### 0.0s - 5.0s: Opening Scene
**Objects Present**: person, car  
**Actions**: Person walking, car entering frame  
**Key Moment** (2.3s): Person stops to wave

### 5.0s - 10.0s: Mid Scene
[...]

## Visual Search Matches

### Match 1: Frame 145 @ 4.83s
**Confidence**: 0.92  
**Description**: Reference object clearly visible in center-right of frame

## Statistics

### Detection Distribution
| Class  | Count | Avg Confidence |
|--------|-------|----------------|
| person | 15    | 0.89           |
| car    | 8     | 0.85           |

### Frame Analysis Coverage
- **Frames Analyzed**: 31/915 (3.4%)
- **Frames with Detections**: 28/31 (90.3%)

## Technical Details
- **LLM Provider**: DeepSeek
- **Model**: deepseek-chat
- **Tokens Used**: 1,247
- **Processing Time**: 2.3 minutes
- **Cache Hits**: 3

---
*Generated by Video Analysis Library v2.0*  
*Timestamp*: 2026-01-18 14:30:22  
*Trace ID*: abc-123-def
```

---

## Security Enhancements v2.0

1. **API Key Management**
   - Environment variable validation
   - Multiple provider key support
   - Secure key storage recommendations

2. **Image Upload Validation**
   - File type verification (PNG, JPG, WEBP)
   - Size limits (max 20MB)
   - Path traversal prevention

3. **Provider Rate Limiting**
   - Per-provider rate limit tracking
   - Automatic backoff strategies
   - Fallback provider switching

---

## Performance Targets v2.0

### Visual Object Search
- Image preprocessing: < 100ms
- Per-frame comparison: < 500ms (with vision API)
- Total search (100 frames): < 60 seconds

### Report Generation
- Markdown formatting: < 200ms
- Statistics computation: < 100ms
- File writing: < 50ms

### Multi-Provider LLM
- Provider switching: < 50ms
- API call overhead: < 100ms
- Token counting: < 10ms

---

## Configuration Examples

### DeepSeek Configuration
```python
llm_config = {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

### Visual Search Configuration
```python
search_config = {
    "provider": "claude",
    "model": "claude-3-sonnet-20240229",
    "similarity_threshold": 0.75,
    "max_matches": 10,
    "description_prompt": "Describe this object in detail"
}
```

### Enhanced Pipeline Usage
```python
pipeline = VideoAnalysisPipeline()

result = pipeline.process_video(
    video_path="video.mp4",
    reference_image_path="search_object.jpg",  # NEW
    llm_config={
        "provider": "deepseek",  # NEW
        "fallback_provider": "claude"  # NEW
    },
    report_format="markdown",  # Generates report.md
    enable_visual_search=True  # NEW
)
```

---

## Backward Compatibility

All v1.0 code remains functional:
- Default provider: Claude (existing behavior)
- report_format="json" still supported
- Visual search is optional (disabled by default)
- All existing tests pass without modification

---

## Migration Guide (v1.0 → v2.0)

### Minimal Changes
```python
# v1.0 code (still works)
result = pipeline.process_video("video.mp4")

# v2.0 with new features
result = pipeline.process_video(
    video_path="video.mp4",
    llm_config={"provider": "deepseek"},  # Optional
    report_format="markdown"  # Now generates report.md
)
```

### New Features Usage
```python
# Visual object search
result = pipeline.process_video(
    video_path="video.mp4",
    reference_image_path="find_this.jpg",
    enable_visual_search=True
)

# Access visual search results
for match in result['visual_search_results']:
    print(f"Found at frame {match['frame_index']} "
          f"with confidence {match['confidence']}")
```

---

## Testing Requirements v2.0

### New Test Cases Required

1. **DeepSeek Integration Tests**
   - API authentication
   - Request/response handling
   - Error handling
   - Fallback mechanism

2. **Visual Search Tests**
   - Reference image loading
   - Frame comparison accuracy
   - Match detection
   - Confidence scoring

3. **Report Generation Tests**
   - Markdown formatting
   - Section completeness
   - Statistics accuracy
   - File writing

4. **Integration Tests**
   - End-to-end with DeepSeek
   - End-to-end with visual search
   - Multi-provider fallback
   - Complete report.md generation

**Total New Tests**: 20+ additional test cases

---

## Implementation Priority

### Phase 1 (Core Features)
1. ✅ Multi-provider LLM support
2. ✅ DeepSeek integration
3. ✅ Enhanced report.md generation

### Phase 2 (Advanced Features)
4. ✅ Visual object search component
5. ✅ Reference image processing
6. ✅ LLM-powered visual matching

### Phase 3 (Polish)
7. ✅ Comprehensive testing
8. ✅ Documentation updates
9. ✅ Performance optimization

---

**Version**: 2.0.0  
**Status**: Ready for Implementation  
**Backward Compatible**: Yes  
**Breaking Changes**: None
