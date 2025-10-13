import cv2
import numpy as np
import requests
import os
from zone_detector import ZoneDetector
class IntrusionDetector:
    def __init__(self, zone_coords, zone_type, confidence_threshold=0.5):
        """
        Initialize the intrusion detection system
        
        Args:
            zone_coords: List of (x, y) tuples defining the restricted zone
            zone_type: Either 'line' or 'polygon'
            confidence_threshold: Minimum confidence for person detection
        """
        self.zone_detector = ZoneDetector(zone_coords, zone_type)
        self.confidence_threshold = confidence_threshold
        self.intrusion_count = 0
        
        # Initialize OpenCV DNN model
        self.model_loaded = False
        self.net = None
        self.output_layers = None
        self.load_yolo_model()
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
        
        # Colors for visualization
        self.colors = {
            'person': (0, 255, 0),      # Green for person detection
            'intrusion': (0, 0, 255),   # Red for intrusion
            'zone': (255, 255, 0),      # Yellow for zone boundary
            'text': (255, 255, 255)     # White for text
        }
    
    def load_yolo_model(self):
        """
        Initialize motion detection-based person detection
        """
        # Initialize background subtractor for motion detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.model_loaded = True
        print("Motion detection system initialized successfully!")
        
    def detect_persons(self, frame):
        """
        Detect moving objects (persons) in the frame using motion detection
        
        Args:
            frame: Input video frame
            
        Returns:
            List of person bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        persons = []
        
        if not self.model_loaded:
            return persons
        
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours to identify potential persons
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter based on area (assuming person-sized objects)
            if area > 500 and area < 50000:  # Adjust these thresholds as needed
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on aspect ratio (height should be greater than width for persons)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 1.0 and h > 50:  # Reasonable person proportions
                    # Calculate confidence based on area and aspect ratio
                    confidence = min(0.9, area / 5000.0)  # Normalize to 0-0.9 range
                    
                    if confidence >= self.confidence_threshold:
                        persons.append((x, y, x + w, y + h, confidence))
        
        return persons
    
    def get_person_center(self, bbox):
        """
        Get the center point of a person's bounding box
        
        Args:
            bbox: Bounding box tuple (x1, y1, x2, y2, confidence)
            
        Returns:
            Tuple (center_x, center_y)
        """
        x1, y1, x2, y2, _ = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)
    
    def draw_detections(self, frame, persons, intrusions):
        """
        Draw person detections and intrusion warnings on the frame
        
        Args:
            frame: Input frame
            persons: List of person bounding boxes
            intrusions: List of intrusion bounding boxes
        """
        # Draw zone boundary
        self.zone_detector.draw_zone(frame)
        
        # Draw person detections
        for person in persons:
            x1, y1, x2, y2, confidence = person
            
            # Check if this person is in intrusion list
            is_intrusion = any(
                abs(x1 - ix1) < 10 and abs(y1 - iy1) < 10 
                for ix1, iy1, ix2, iy2, _ in intrusions
            )
            
            color = self.colors['intrusion'] if is_intrusion else self.colors['person']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
            
            # Draw center point
            center = self.get_person_center(person)
            cv2.circle(frame, center, 5, color, -1)
        
        # Draw intrusion warning if any intrusions detected
        if intrusions:
            self.draw_intrusion_warning(frame)
    
    def draw_intrusion_warning(self, frame):
        """
        Draw intrusion warning text on the frame
        
        Args:
            frame: Input frame
        """
        warning_text = "INTRUSION DETECTED!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        # Get text size
        text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[0]
        
        # Calculate position (top center)
        frame_height, frame_width = frame.shape[:2]
        text_x = (frame_width - text_size[0]) // 2
        text_y = 50
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     self.colors['intrusion'], -1)
        
        # Draw warning text
        cv2.putText(frame, warning_text, (text_x, text_y), 
                   font, font_scale, self.colors['text'], thickness)
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """
        Process entire video for intrusion detection
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Total number of intrusions detected
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_intrusions = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect persons in current frame
                persons = self.detect_persons(frame)
                
                # Check for intrusions
                intrusions = []
                for person in persons:
                    center = self.get_person_center(person)
                    if self.zone_detector.is_point_in_zone(center):
                        intrusions.append(person)
                        total_intrusions += 1
                
                # Draw detections and warnings
                self.draw_detections(frame, persons, intrusions)
                
                # Write frame to output video
                out.write(frame)
                
                frame_count += 1
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress)
        
        finally:
            cap.release()
            out.release()
        
        return total_intrusions
