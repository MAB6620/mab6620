# -*- coding:utf8 -*-
# !/usr/bin/env python3
# Copyright 2025 Octadero.
# Author Volodymyr Pavliukevych

"""This is a Vehicle Tracker usage example
This script initializes the vehicle tracker, processes a video file,
and outputs the tracked vehicles with their IDs.
"""

import os
import random
import tyro
import cv2
from datetime import datetime
from vehicle_tracker import VehicleTracker, TrackingParams, TrackingMode
from dataclasses import dataclass, fields
import numpy as np

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    tag: str | None = None
    """the name of this experiment"""
    dataset_folder_name: str = "datasets/evaluation"
    """Path to the dataset folder"""
    video_file_to_process: str = "car1.mp4"
    """Path to the video file to process"""
    video_output_file_name: str = "output_video.mp4"
    """Path to the output video file"""
    video_output_folder_name: str = "output"
    """Path to the output video folder"""
    draw_fps: bool = True
    """Whether to draw FPS on the video"""
    different_colors: bool = True
    """Whether to use different colors for each vehicle"""
    scale_factor: float = 1.0
    """Scale factor for resizing the video frames"""

    def overridden_args(self) -> dict:
        default_instance = Args()
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != getattr(default_instance, f.name)
        }    


def main(args: Args):
    print(f"Overridden args: {args.overridden_args()}")
    # Initialize the tracker
    tracker = VehicleTracker(params={
        TrackingParams.MODE.key : TrackingMode.BALANCED
    })

    src_file_path = os.path.join(args.dataset_folder_name, args.video_file_to_process)
    if not os.path.exists(src_file_path):
        print(f"File {src_file_path} does not exist.")
        return

    
    src_file_name = os.path.splitext(args.video_file_to_process)[0]

    output_file_path = os.path.join(args.video_output_folder_name, f"{src_file_name}_{args.video_output_file_name}")
    if os.path.exists(output_file_path):
        print(f"Output file: '{output_file_path}' will be overwritten.")
        

    # Load video
    video_capture = cv2.VideoCapture(src_file_path)
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get video properties for output writer
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    resized_shape = (int(args.scale_factor * frame_width), int(args.scale_factor * frame_height))
    out = cv2.VideoWriter(output_file_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, 
                          resized_shape)

    frame_index = 0
    number_objects = 0
    while video_capture.isOpened():
        print(f"Processing frame {frame_index}/{total_frame_count}")
        frame_index += 1
        ret, frame = video_capture.read()
        if not ret:
            break
        # Resize the frame if needed
        frame = cv2.resize(frame, resized_shape)
        # # Convert the frame to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = datetime.now().timestamp()
        # Update the tracker with the current frame
        tracker.update(frame)
        vehicles = tracker.get_vehicles()
        end_time = datetime.now().timestamp()

        if args.draw_fps:
            
            fps = fps * 0.45 + (1 / (end_time - start_time) + 1e-6) * 0.5
            cv2.putText(frame, f'[{frame_index}] FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for vehicle in vehicles:
            number_objects = number_objects if number_objects > vehicle.id else vehicle.id
            x1, y1, x2, y2 = vehicle.bbox
            
            if args.different_colors:
                random.seed(int(vehicle.id)+10)

            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"vehicle: {vehicle.id}, conf: {vehicle.confidence:0.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # Write the frame to the output video
        out.write(frame)

        # Optionally display it
        # cv2.imshow("Tracking", frame)
        # if cv2.waitKey(600) == 27:  # Esc to quit
        #     break
    print(f"Result FPS: {fps:.2f} Total objects: {number_objects}")
    # Release everything
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)