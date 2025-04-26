# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Octadero.
# Author Volodymyr Pavliukevych

"""This is a Vehicle Tracker representation
"""

import cv2
import os
import numpy as np
from enum import StrEnum
from ultralytics import YOLO
from ultralytics.engine.results import Results
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackingMode(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    SLOW_AND_ACCURATE = "slow_and_accurate"

class TrackingParams(StrEnum):
    MODE = "mode"
    MODEL_FOLDER = "model_folder"
    TRACKING_MAX_AGE = "tracking_max_age"
    CUSTOM_MODEL_NAME = "custom_model_name"

    TRACKING_MODE = "tracking_mode"
    TRACKING_CONFIDENCE = "tracking_confidence"
    TRACKING_NMS = "tracking_nms"
    VERBOSE = "verbose"
    TRACKING_MAX_DETECTIONS = "tracking_max_detections"
    # TRACKING_MIN_HITS = "tracking_min_hits"
    # TRACKING_IOT = "tracking_iot"
    # TRACKING_MAX_DIST = "tracking_max_dist"
    # TRACKING_N_INIT = "tracking_n_init"
    # TRACKING_MAX_COSINE_DISTANCE = "tracking_max_cosine_distance"
    # TRACKING_NMS_MAX_DETECTIONS = "tracking_nms_max_detections"
    # TRACKING_NMS_IOU = "tracking_nms_iou"

    def default(self) -> float:
        if self == TrackingParams.MODE:
            return TrackingMode.SLOW_AND_ACCURATE
        elif self == TrackingParams.MODEL_FOLDER:
            return "models"
        elif self == TrackingParams.TRACKING_MAX_AGE:
            return 30
        elif self == TrackingParams.CUSTOM_MODEL_NAME:
            return None
        elif self == TrackingParams.TRACKING_CONFIDENCE:
            return 0.5
        elif self == TrackingParams.TRACKING_NMS:
            return 0.5
        elif self == TrackingParams.VERBOSE:
            return False
        else:
            raise ValueError(f"Unknown parameter: {self}")

    def key(self) -> str:
        return self.value
    
    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))      


class Vehicle:
    def __init__(self, id: int, bbox: np.ndarray, confidence: float):
        self.lifespan = 2
        self.id = id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.history = [bbox]
        if confidence is not None:
            self.confidence = confidence
        else:
            self.confidence = 0.0
        
    def update(self, bbox: np.ndarray, confidence: float):
        self.bbox = bbox
        self.lifespan += 1
        if confidence is not None:
            self.confidence = confidence
        else:
            self.confidence = 0.0
        self.history.append(bbox)

    def update_lifespan(self) -> bool:
        self.lifespan -= 1
        if self.lifespan <= 0:
            return False
        return True

class VehicleTracker:
    def __init__(self, params:dict = None):
        self.params = params or {}
        mode = TrackingMode(self.params.get(TrackingParams.MODE.key, TrackingMode.BALANCED))
        model_folder = self.params.get(TrackingParams.MODEL_FOLDER.key, "models")

        # Choose YOLO model based on mode
        if mode == TrackingMode.FAST:
            model_name = "yolov8n.pt"
        elif mode == TrackingMode.ACCURATE:
            model_name = "yolov8m.pt"
        elif mode == TrackingMode.SLOW_AND_ACCURATE:
            model_name = "yolov8l.pt"
        else:  # BALANCED
            model_name = "yolov8s.pt"
        
        # Check for custom model name
        # If a custom model name is provided, use it instead of the default
        # model name. Ensure the custom model exists in the specified folder.
        # If the custom model name is not a string or is empty, fall back to the default model name.
        custom_model_name = self.params.get(TrackingParams.CUSTOM_MODEL_NAME.key)
        if isinstance(custom_model_name, str) and custom_model_name:
            if os.path.exists(os.path.join(model_folder, custom_model_name)):
                model_name = custom_model_name

        model_path = os.path.join(model_folder, model_name)
        verbose = self.params.get(TrackingParams.VERBOSE.key, False)

        self.detector = YOLO(model_path, verbose=verbose)
        self.detector.fuse()
        self.detector.half()
        self.detector.conf = self.params.get(TrackingParams.TRACKING_CONFIDENCE.key, 0.5)
        self.detector.iou = self.params.get(TrackingParams.TRACKING_NMS.key, 0.5)
        self.detector.agnostic_nms = True
        self.detector.classes = [2, 7]
        self.detector.max_det = 1000
        self.detector.amp = True
        self.detector.tracker = None
        

        tracking_max_age = self.params.get(TrackingParams.TRACKING_MAX_AGE.key, 30)
        self.tracker = DeepSort(max_age=tracking_max_age, 
                                max_iou_distance=0.4, 
                                n_init=2, 
                                nn_budget=None,  
                                nms_max_overlap=0.5, 
                                max_cosine_distance=0.5,
                                embedder="mobilenet")
        self.vehicles = {}

    def update(self, frame: np.ndarray) -> None:
        """Update the tracker with a new frame."""

        results: Results = self.detector.track(
            frame,  
            tracker="models/trackers/botsort.yaml",
            show=False,
            persist=True,
            conf=self.params.get(TrackingParams.TRACKING_CONFIDENCE.key, 0.45),
            iou=self.params.get(TrackingParams.TRACKING_NMS.key, 0.05),
            agnostic_nms=True,
            classes=[2, 7],
            max_det=10,
            amp=True)
                
        # Loop over frames
        for result in results:
            # Each `result` is a `Results` object for one frame
            boxes = result.boxes

            if boxes is not None:
                # .xyxy: (x1, y1, x2, y2), .conf: confidence, .cls: class ID, .id: object ID
                bboxes = boxes.xyxy.cpu().numpy()       # shape (N, 4)
                confidences = boxes.conf.cpu().numpy()  # shape (N,)
                class_ids = boxes.cls.cpu().numpy()     # shape (N,)
                object_ids = boxes.id.cpu().numpy() if boxes.id is not None else None  # shape (N,)

                # Example: print each detection
                for i in range(len(bboxes)):
                    x1, y1, x2, y2 = map(int, bboxes[i])
                    bbox = tuple(map(int, bboxes[i]))
                    conf = confidences[i]
                    cls = int(class_ids[i])
                    obj_id = int(object_ids[i]) if object_ids is not None else -1
                    print(f"Object ID: {obj_id}, Class: {cls}, Conf: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
                    if obj_id in self.vehicles:
                        self.vehicles[obj_id].update(bbox, conf)
                    else:
                        self.vehicles[obj_id] = Vehicle(obj_id, bbox, confidence=conf)


        # detections = []

        # for box in results.boxes.data:
        #     x1, y1, x2, y2, conf, cls = box
        #     if int(cls) == 2 or int(cls) == 7:  # Car or truck
        #         detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, 'vehicle'))

        # tracks = self.tracker.update_tracks(detections, frame=frame)
        # for track in tracks:
        #     if not track.is_confirmed():
        #         continue
        #     track_id = track.track_id
        #     ltrb = track.to_ltrb()
        #     bbox = tuple(map(int, ltrb))

        #     if track_id in self.vehicles:
        #         self.vehicles[track_id].update(bbox, track.det_conf)
        #     else:
        #         self.vehicles[track_id] = Vehicle(track_id, bbox, confidence=track.det_conf)
            
            # Remove vehicles that have not been updated for a while
            for vehicle in list(self.vehicles.values()):
                if not vehicle.update_lifespan():
                    del self.vehicles[vehicle.id]
        

    def get_vehicles(self) -> list[Vehicle]:
        """Get the list of tracked vehicles."""
        return list(self.vehicles.values())
