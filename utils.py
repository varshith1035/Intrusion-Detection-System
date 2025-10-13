mport base64
import os
import numpy as np
import cv2
def validate_coordinates(zone_coords, frame_width, frame_height):
    """
    Validate that zone coordinates are within frame boundaries
    
    Args:
        zone_coords: List of (x, y) tuples
        frame_width: Width of video frame
        frame_height: Height of video frame
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not zone_coords:
        return False, "No coordinates provided"
    
    if len(zone_coords) < 2:
        return False, "At least 2 coordinates required"
    
    for i, (x, y) in enumerate(zone_coords):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False, f"Point {i+1}: Coordinates must be numeric"
        
        if x < 0 or x >= frame_width:
            return False, f"Point {i+1}: X coordinate {x} is outside frame width (0-{frame_width-1})"
        
        if y < 0 or y >= frame_height:
            return False, f"Point {i+1}: Y coordinate {y} is outside frame height (0-{frame_height-1})"
    
    return True, "Valid coordinates"
def create_download_link(file_path, link_text="Download File"):
    """
    Create a download link for a file
    
    Args:
        file_path: Path to the file
        link_text: Text to display for the link
        
    Returns:
        HTML string for download link
    """
    if not os.path.exists(file_path):
        return "File not found"
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href
def get_video_info(video_path):
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = float(info['frame_count']) / float(info['fps'])
    
    cap.release()
    return info
# Additional utility functions for geometric calculations
def resize_frame(frame, max_width=800, max_height=600):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return frame
def format_time(seconds):
    """Format seconds into HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"
def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
