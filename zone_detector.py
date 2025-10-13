import cv2
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
class ZoneDetector:
    def __init__(self, zone_coords, zone_type):
        """
        Initialize zone detector
        
        Args:
            zone_coords: List of (x, y) tuples defining the zone
            zone_type: Either 'line' or 'polygon'
        """
        self.zone_coords = zone_coords
        self.zone_type = zone_type.lower()
        
        # Create shapely geometry object
        if self.zone_type == 'line':
            if len(zone_coords) < 2:
                raise ValueError("Line zone requires at least 2 points")
            self.zone_geometry = LineString(zone_coords)
            # For line crossing detection, we'll use a buffer
            self.zone_buffer = self.zone_geometry.buffer(20)  # 20 pixel buffer
        elif self.zone_type == 'polygon':
            if len(zone_coords) < 3:
                raise ValueError("Polygon zone requires at least 3 points")
            self.zone_geometry = Polygon(zone_coords)
        else:
            raise ValueError("Zone type must be 'line' or 'polygon'")
    
    def is_point_in_zone(self, point):
        """
        Check if a point is inside the restricted zone
        
        Args:
            point: Tuple (x, y) representing the point to check
            
        Returns:
            Boolean indicating if point is in zone
        """
        shapely_point = Point(point)
        
        if self.zone_type == 'line':
            # For line zones, check if point is within buffer distance
            return self.zone_buffer.contains(shapely_point)
        else:  # polygon
            return self.zone_geometry.contains(shapely_point)
    
    def get_distance_to_zone(self, point):
        """
        Get the distance from a point to the zone boundary
        
        Args:
            point: Tuple (x, y) representing the point
            
        Returns:
            Distance to zone boundary
        """
        shapely_point = Point(point)
        return shapely_point.distance(self.zone_geometry)
    
    def draw_zone(self, frame):
        """
        Draw the zone boundary on the frame
        
        Args:
            frame: OpenCV frame to draw on
        """
        color = (0, 255, 255)  # Yellow color for zone
        thickness = 3
        
        if self.zone_type == 'line':
            # Draw line
            if len(self.zone_coords) >= 2:
                pt1 = tuple(map(int, self.zone_coords[0]))
                pt2 = tuple(map(int, self.zone_coords[1]))
                cv2.line(frame, pt1, pt2, color, thickness)
                
                # Draw end points
                cv2.circle(frame, pt1, 8, color, -1)
                cv2.circle(frame, pt2, 8, color, -1)
                
                # Add zone label
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                cv2.putText(frame, "RESTRICTED ZONE", (mid_x - 80, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        else:  # polygon
            # Draw polygon
            points = np.array(self.zone_coords, dtype=np.int32)
            cv2.polylines(frame, [points], True, color, thickness)
            
            # Fill polygon with semi-transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
            
            # Draw corner points
            for point in self.zone_coords:
                pt = tuple(map(int, point))
                cv2.circle(frame, pt, 6, color, -1)
            
            # Add zone label at centroid
            if len(self.zone_coords) >= 3:
                centroid_x = int(sum(p[0] for p in self.zone_coords) / len(self.zone_coords))
                centroid_y = int(sum(p[1] for p in self.zone_coords) / len(self.zone_coords))
                cv2.putText(frame, "RESTRICTED ZONE", (centroid_x - 80, centroid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def is_crossing_line(self, prev_point, curr_point):
        """
        Check if movement from prev_point to curr_point crosses the line zone
        Only applicable for line zones
        
        Args:
            prev_point: Previous position (x, y)
            curr_point: Current position (x, y)
            
        Returns:
            Boolean indicating if line was crossed
        """
        if self.zone_type != 'line':
            return False
        
        if prev_point is None or curr_point is None:
            return False
        
        # Create line segment from movement
        movement_line = LineString([prev_point, curr_point])
        
        # Check if movement line intersects zone line
        return self.zone_geometry.intersects(movement_line)
    
    def get_zone_info(self):
        """
        Get information about the zone
        
        Returns:
            Dictionary with zone information
        """
        info = {
            'type': self.zone_type,
            'coordinates': self.zone_coords,
            'num_points': len(self.zone_coords)
        }
        
        if self.zone_type == 'polygon':
            info['area'] = self.zone_geometry.area
        else:  # line
            info['length'] = self.zone_geometry.length
        
        return info
