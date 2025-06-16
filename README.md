
```
params = {
    "mode": "fast",
    "tracking_confidence": 0.6,
    "tracking_max_age": 20
}
```
```
tracker = VehicleTracker(params=params)
```

**Notes**

The project uses `yolov8n.pt` for `Fast` mode, `yolov8s.pt` for `Balanced`, `yolov8m.pt` for `Accurate`, and `yolov8l.pt` for `Slow-and-Accurate`. Larger models (e.g., `yolov8l.pt`) are more accurate but slower.

The deep-sort-realtime library is used for tracking, with parameters tuned for vehicle tracking.
Ensure sufficient disk space for output videos, as they can be large depending on input video size and duration.
For GPU acceleration, ensure CUDA and cuDNN are installed and compatible with your system.


### Improvements, ToDO:
- Use a custom-trained model for better detection accuracy.
- Implement temporal smoothing to reduce jitter.
- Add motion models to improve prediction during occlusion.
- Make GPU computation possible

## Preparing Dataset

```
ffmpeg -i datasets/evaluation/car1.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_1_image_%03d.jpg
ffmpeg -i datasets/evaluation/car2.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_2_image_%03d.jpg
ffmpeg -i datasets/evaluation/car3.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_3_image_%03d.jpg
ffmpeg -i datasets/evaluation/car4.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_4_image_%03d.jpg
ffmpeg -i datasets/evaluation/car5.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_5_image_%03d.jpg
ffmpeg -i datasets/evaluation/car6.mp4 -vf "fps=5,scale=trunc(iw/2)*2:trunc(ih/2)*2" datasets/train/car_6_image_%03d.jpg
```

**License**
- Copyright © 2025 Octadero. All rights reserved.
- Acknowledgments

- Ultralytics YOLOv8 for object detection.
- DeepSort-Realtime for object tracking.
