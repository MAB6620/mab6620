
### Vehicle tracker

**Approach**:
- **Detection**: YOLOv8 (small model for real-time use).
- **Tracking**: Deep SORT for appearance-based tracking and occlusion recovery.
- **API**: Simple object-oriented design for modular usage.

**Limitations**:
- May struggle with very fast or heavily occluded vehicles.
- Limited to classes explicitly filtered (cars and trucks).
- Runs in real time only on moderate resolutions (~720p).

**Improvements**:
- Use a custom-trained model for better detection accuracy.
- Implement temporal smoothing to reduce jitter.
- Add motion models to improve prediction during occlusion.
- Make GPU computation possible
- Resize Video