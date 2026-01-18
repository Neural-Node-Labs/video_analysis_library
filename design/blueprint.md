# Video Analysis Library - Component Decomposition Blueprint

## System Overview
AI-powered video analysis system that extracts frames, detects objects/movements, and generates intelligent summaries using LLM integration.

---

## Component Decomposition Table

### 1. VideoLoaderComponent
**Logical Function**: Load and validate video files from various sources
**IN Schema**:
```json
{
  "video_path": "string",
  "validation_mode": "enum[strict, lenient]"
}
```
**OUT Schema**:
```json
{
  "video_handle": "object",
  "metadata": {
    "duration_sec": "float",
    "fps": "float",
    "resolution": "tuple[int, int]",
    "codec": "string",
    "total_frames": "int"
  },
  "status": "enum[success, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[FILE_NOT_FOUND, INVALID_FORMAT, CORRUPTED_FILE]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- video_load_start
- video_validation_complete
- metadata_extracted

**Adaptor Requirements**: None (entry point)

---

### 2. FrameExtractorComponent
**Logical Function**: Extract frames from video at specified intervals
**IN Schema**:
```json
{
  "video_handle": "object",
  "extraction_mode": "enum[interval, keyframe, all]",
  "fps_target": "float|null",
  "frame_interval_sec": "float|null"
}
```
**OUT Schema**:
```json
{
  "frames": "list[numpy.ndarray]",
  "timestamps": "list[float]",
  "frame_count": "int",
  "status": "enum[success, partial, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[EXTRACTION_FAILED, MEMORY_ERROR, INVALID_HANDLE]",
  "message": "string",
  "failed_at_frame": "int|null",
  "trace_id": "uuid"
}
```
**Trace Points**:
- extraction_start
- frame_batch_extracted (every 100 frames)
- extraction_complete

**Adaptor Requirements**: 
- VideoLoaderAdaptor (receives video_handle)

---

### 3. FramePreprocessorComponent
**Logical Function**: Normalize and prepare frames for AI analysis
**IN Schema**:
```json
{
  "frames": "list[numpy.ndarray]",
  "target_size": "tuple[int, int]|null",
  "normalization": "enum[standard, minmax, none]",
  "color_space": "enum[RGB, BGR, GRAYSCALE]"
}
```
**OUT Schema**:
```json
{
  "processed_frames": "list[numpy.ndarray]",
  "preprocessing_stats": {
    "mean": "float",
    "std": "float",
    "resize_applied": "bool"
  },
  "status": "enum[success, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[INVALID_DIMENSIONS, PROCESSING_ERROR]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- preprocessing_start
- normalization_applied
- preprocessing_complete

**Adaptor Requirements**:
- FrameExtractorAdaptor

---

### 4. ObjectDetectionComponent
**Logical Function**: Detect objects in frames using AI models
**IN Schema**:
```json
{
  "frames": "list[numpy.ndarray]",
  "model_type": "enum[yolo, faster-rcnn, ssd]",
  "confidence_threshold": "float",
  "device": "enum[cpu, cuda, mps]"
}
```
**OUT Schema**:
```json
{
  "detections": "list[FrameDetection]",
  "detection_summary": {
    "total_objects": "int",
    "unique_classes": "list[string]",
    "avg_confidence": "float"
  },
  "status": "enum[success, error]"
}

FrameDetection = {
  "frame_index": "int",
  "timestamp": "float",
  "objects": "list[Object]"
}

Object = {
  "class": "string",
  "confidence": "float",
  "bbox": "tuple[int, int, int, int]",
  "object_id": "uuid"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[MODEL_LOAD_FAILED, INFERENCE_ERROR, DEVICE_ERROR]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- model_loaded
- inference_batch_start
- inference_batch_complete
- detection_complete

**Adaptor Requirements**:
- FramePreprocessorAdaptor

---

### 5. MotionAnalysisComponent
**Logical Function**: Analyze movement patterns between consecutive frames
**IN Schema**:
```json
{
  "frames": "list[numpy.ndarray]",
  "detections": "list[FrameDetection]",
  "motion_algorithm": "enum[optical_flow, frame_diff, object_tracking]"
}
```
**OUT Schema**:
```json
{
  "motion_events": "list[MotionEvent]",
  "motion_summary": {
    "total_movements": "int",
    "avg_velocity": "float",
    "motion_hotspots": "list[tuple[int, int]]"
  },
  "status": "enum[success, error]"
}

MotionEvent = {
  "start_frame": "int",
  "end_frame": "int",
  "object_id": "uuid",
  "trajectory": "list[tuple[int, int]]",
  "velocity": "float",
  "direction": "float"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[TRACKING_LOST, ALGORITHM_ERROR]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- motion_analysis_start
- tracking_initialized
- motion_analysis_complete

**Adaptor Requirements**:
- FramePreprocessorAdaptor
- ObjectDetectionAdaptor

---

### 6. LLMAnalysisComponent
**Logical Function**: Generate natural language descriptions using LLM
**IN Schema**:
```json
{
  "detections": "list[FrameDetection]",
  "motion_events": "list[MotionEvent]",
  "video_metadata": "object",
  "analysis_type": "enum[summary, detailed, narrative]",
  "llm_config": {
    "model": "string",
    "temperature": "float",
    "max_tokens": "int"
  }
}
```
**OUT Schema**:
```json
{
  "summary": "string",
  "scene_descriptions": "list[SceneDescription]",
  "key_moments": "list[KeyMoment]",
  "status": "enum[success, error]"
}

SceneDescription = {
  "time_range": "tuple[float, float]",
  "description": "string",
  "objects_present": "list[string]",
  "actions": "list[string]"
}

KeyMoment = {
  "timestamp": "float",
  "description": "string",
  "importance_score": "float"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[LLM_API_ERROR, RATE_LIMIT, CONTEXT_TOO_LONG]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- llm_request_start
- llm_response_received
- analysis_complete

**Adaptor Requirements**:
- ObjectDetectionAdaptor
- MotionAnalysisAdaptor

---

### 7. ReportGeneratorComponent
**Logical Function**: Generate structured reports in various formats
**IN Schema**:
```json
{
  "llm_analysis": "object",
  "detections": "list[FrameDetection]",
  "motion_events": "list[MotionEvent]",
  "report_format": "enum[json, markdown, html, pdf]",
  "include_visualizations": "bool"
}
```
**OUT Schema**:
```json
{
  "report_content": "string|bytes",
  "report_path": "string|null",
  "metadata": {
    "generated_at": "datetime",
    "format": "string",
    "size_bytes": "int"
  },
  "status": "enum[success, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[GENERATION_FAILED, WRITE_ERROR, FORMAT_ERROR]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- report_generation_start
- report_written
- report_complete

**Adaptor Requirements**:
- LLMAnalysisAdaptor
- ObjectDetectionAdaptor
- MotionAnalysisAdaptor

---

### 8. CacheManagerComponent
**Logical Function**: Cache intermediate results for performance
**IN Schema**:
```json
{
  "operation": "enum[get, set, invalidate]",
  "cache_key": "string",
  "data": "object|null",
  "ttl_seconds": "int|null"
}
```
**OUT Schema**:
```json
{
  "cache_hit": "bool",
  "data": "object|null",
  "status": "enum[success, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[CACHE_MISS, SERIALIZATION_ERROR]",
  "message": "string",
  "trace_id": "uuid"
}
```
**Trace Points**:
- cache_access
- cache_write
- cache_invalidation

**Adaptor Requirements**: Used by all components

---

### 9. LoggingComponent
**Logical Function**: Dual-stream logging (system + LLM interaction)
**IN Schema**:
```json
{
  "log_type": "enum[system, llm_interaction]",
  "level": "enum[DEBUG, INFO, WARN, ERROR]",
  "message": "string",
  "metadata": "object",
  "trace_id": "uuid"
}
```
**OUT Schema**:
```json
{
  "logged": "bool",
  "timestamp": "datetime",
  "status": "enum[success, error]"
}
```
**Error Schema**:
```json
{
  "error_code": "enum[WRITE_FAILED, PERMISSION_DENIED]",
  "message": "string"
}
```
**Trace Points**:
- log_written

**Adaptor Requirements**: Used by all components

---

## Adaptor Definitions

### VideoLoaderAdaptor
Transforms VideoLoaderComponent.OUT → FrameExtractorComponent.IN

### FrameExtractorAdaptor
Transforms FrameExtractorComponent.OUT → FramePreprocessorComponent.IN

### FramePreprocessorAdaptor
Transforms FramePreprocessorComponent.OUT → ObjectDetectionComponent.IN + MotionAnalysisComponent.IN

### ObjectDetectionAdaptor
Transforms ObjectDetectionComponent.OUT → MotionAnalysisComponent.IN + LLMAnalysisComponent.IN

### MotionAnalysisAdaptor
Transforms MotionAnalysisComponent.OUT → LLMAnalysisComponent.IN

### LLMAnalysisAdaptor
Transforms LLMAnalysisComponent.OUT → ReportGeneratorComponent.IN

---

## Execution Flow

```
VideoFile → VideoLoader → FrameExtractor → FramePreprocessor
                                                  ↓
                                          ObjectDetection
                                                  ↓
                                          MotionAnalysis
                                                  ↓
                                           LLMAnalysis
                                                  ↓
                                         ReportGenerator → Output
                                         
[CacheManager + LoggingComponent support all stages]
```

---

## Security Controls

1. **Path Traversal Prevention**: VideoLoaderComponent validates file paths
2. **Input Sanitization**: All components validate schema before processing
3. **Resource Limits**: Memory caps on frame extraction (max 1000 frames/batch)
4. **Error Isolation**: Transaction rollback on component failures
5. **API Key Protection**: LLMAnalysisComponent uses environment variables

---

## Performance Targets

- VideoLoader: < 100ms
- FrameExtractor: < 2ms/frame
- ObjectDetection: < 50ms/frame (GPU), < 200ms/frame (CPU)
- MotionAnalysis: < 30ms/frame
- LLMAnalysis: < 3s (API dependent)
- ReportGenerator: < 500ms

**Total Pipeline**: < 5 minutes for 1-minute video @ 30fps
