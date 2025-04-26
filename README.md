
### Vehicle Tracker
This project implements a vehicle tracking system using YOLOv8 for object detection and DeepSort for tracking. It processes video files to detect and track vehicles (cars and trucks), outputting a video with bounding boxes and IDs for each tracked vehicle.

**Features**

 - Supports multiple tracking modes: Fast, Balanced, Accurate, and Slow-and-Accurate.
 - Configurable parameters for detection confidence, non-max suppression, and tracking age.
- Processes video files and saves output with annotated vehicle bounding boxes and IDs.
- Optional FPS display and per-vehicle color differentiation in output videos.
- Modular design with customizable YOLO model selection.

### Installation
**Prerequisites**

Python 3.8 or higher
A system with sufficient GPU memory for YOLOv8 models (recommended for performance)
Video files for processing (e.g., .mp4 files in the `dataset/evaluation` folder)

**Setup**

Clone the Repository:
```
git clone https://github.com/VolodymyrPavliukevych/dragonfly
cd dragonfly
```

Create a Virtual Environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies**: 

 - `opencv-python`
 - `numpy`
 - `ultralytics`
 - `deep-sort-realtime`
 - `tyro`

**Install them using**:
```
pip3 install -r requirements.txt
```

Download YOLOv8 Models: The project uses pretrained YOLOv8 models (`yolov8n.pt`, `yolov8s.pt`, `yolov8l.pt`). These should be placed in the `models/` folder. 
You can download them from the Ultralytics YOLOv8 repository or use the provided models if already included.

Prepare Video Files: Place your input video files (e.g., `car1.mp4`, `car2.mp4`) in the `dataset/evaluation/` folder. The provided dataset includes sample videos for testing.

### Git Large File System
Project includes git lfs solution for storing models artifacts and video files.
Make sure that git LFS installed on your PC.
```
brew install git-lfs
git lfs install
git lfs track "*.pt"
git lfs track "*.mp4"
git add models/*.pt
git add dataset/evaluation/*.mp4

```

Output should be like:
```
Pushing to github.com:VolodymyrPavliukevych/dragonfly.git
To github.com:VolodymyrPavliukevych/dragonfly.git
   625e395..3664114  main -> main
updating local tracking ref 'refs/remotes/origin/main'
Uploading LFS objects: 100% (9/9), 788 MB | 12 MB/s, done.
``` 


**Directory Structure**:
```
dragonfly/
├── dataset/
│   └── evaluation/
│       ├── car1.mp4
│       ├── car2.mp4
│       └── ...
├── models/
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── yolov8l.pt
│   └── trackers/
│       ├── botsort.yaml
│       └── bytetrack.yaml
├── output/
│   ├── car1_output_video.mp4
│   └── ...
├── requirements.txt
├── vehicle_tracker.py
└── vehicle_tracker_luncher.py
└── README.md
```

### Usage

**Run the Tracker**: Use the `vehicle_tracker_luncher.py` script to process a video file. Example command:

```
python vehicle_tracker_luncher.py --video_file_to_process car1.mp4
```

This will process `car1.mp4` and save the output to `output/car1_output_video.mp4`.

**Command-Line Arguments**: The script supports several configurable parameters:

`--exp_name` : Experiment name (default: script name).
`--tag` : Optional tag for the experiment.
`--dataset_folder_name` : Path to the dataset folder (default: dataset/evaluation).
`--video_file_to_process` : Input video file name (default: car1.mp4).
`--video_output_file_name` : Output video file name (default: output_video.mp4).
`--video_output_folder_name` : Output folder (default: output).
`--draw_fps` : Display FPS on output video (default: True).
`--different_colors` : Use unique colors for each vehicle (default: True).
`--scale_factor` : Resize factor for video frames (default: 1.0).

**Example with custom parameters**:
```
python vehicle_tracker_luncher.py --video_file_to_process car2.mp4 --scale_factor 0.5 --draw_fps False
```

**Output**: The processed video will be saved in the output/ folder with bounding boxes, vehicle IDs, and confidence scores. If --draw_fps is enabled, the FPS and object count will be displayed on the video.


**Configuration**
The `vehicle_tracker.py` script allows customization through a params dictionary. Key parameters include:

`mode`: Tracking mode (fast, balanced, accurate, slow_and_accurate).
`model_folder`: Folder containing YOLO models (default: models).
`tracking_max_age`: Maximum age for tracking objects (default: 30).
`custom_model_name`: Optional custom YOLO model file.
`tracking_confidence`: Detection confidence threshold (default: 0.5).
`tracking_nms`: Non-max suppression threshold (default: 0.5).
`verbose`: Enable verbose logging (default: False).

Example of passing custom parameters in code:
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


**License**
Copyright © 2025 Octadero. All rights reserved.
Acknowledgments

Ultralytics YOLOv8 for object detection.
DeepSort-Realtime for object tracking.
