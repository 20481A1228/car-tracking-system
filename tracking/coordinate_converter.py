# tracking/coordinate_converter.py - ENHANCED COORDINATE CONVERSION SYSTEM

import cv2
import numpy as np
import os
import math

class CoordinateConverter:
    """Enhanced coordinate converter with K-nearest neighbors and polygon validation."""
    
    def __init__(self, image_path=None, NBlot_coords=None, geo_coords=None, image_width=640, image_height=360):
        # Set image dimensions
        self.image_width = image_width
        self.image_height = image_height
        
        # Load image if provided
        if image_path and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image_height, self.image_width = self.image.shape[:2]
            else:
                self.image = None
                print(f"Warning: Could not load image: {image_path}")
        else:
            self.image = None
        
        # Set coordinates
        self.NBlot_coords = NBlot_coords or []
        self.NBlot_geo = geo_coords or []
        
        # Convert YOLO coordinates to pixel coordinates
        if self.NBlot_coords:
            self.NBlot_pixels = self.yolo_to_pixel_coords(self.NBlot_coords)
        else:
            self.NBlot_pixels = []
        
        # Default K for KNN
        self.k = min(4, len(self.NBlot_geo)) if self.NBlot_geo else 4
        
        print(f"✅ CoordinateConverter initialized:")
        print(f"   - Image dimensions: {self.image_width}x{self.image_height}")
        print(f"   - Polygon points: {len(self.NBlot_coords)}")
        print(f"   - Reference coordinates: {len(self.NBlot_geo)}")
        print(f"   - K-neighbors: {self.k}")
    
    def yolo_to_pixel_coords(self, yolo_coords):
        """Convert YOLO format coordinates to pixel coordinates."""
        pixel_coords = []
        for x, y in yolo_coords:
            pixel_x = int(x * self.image_width)
            pixel_y = int(y * self.image_height)
            pixel_coords.append((pixel_x, pixel_y))
        return pixel_coords
    
    def pixel_to_yolo_coords(self, pixel_coords):
        """Convert pixel coordinates to YOLO format."""
        yolo_coords = []
        for x, y in pixel_coords:
            yolo_x = x / self.image_width
            yolo_y = y / self.image_height
            yolo_coords.append((yolo_x, yolo_y))
        return yolo_coords
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
        if not polygon or len(polygon) < 3:
            return False
            
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
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def compute_geolocation_knn(self, pixel_x, pixel_y, k=None):
        """
        Compute approximate geolocation using K-nearest neighbors with polygon validation.
        
        Args:
            pixel_x, pixel_y: The pixel coordinates to convert
            k: Number of nearest neighbors to use (defaults to self.k)
        
        Returns:
            (latitude, longitude) or (None, None) if outside polygon or conversion fails
        """
        if k is None:
            k = self.k
            
        # Ensure we have reference points
        if not self.NBlot_pixels or not self.NBlot_geo:
            print("Warning: No reference points available for coordinate conversion")
            return None, None
        
        k = min(k, len(self.NBlot_geo))
        
        # First check if point is inside the polygon (CRITICAL for accuracy)
        if not self.point_in_polygon((pixel_x, pixel_y), self.NBlot_pixels):
            return None, None
        
        # Calculate distances to all reference points
        distances = []
        for i, (px, py) in enumerate(self.NBlot_pixels):
            dist = math.sqrt((pixel_x - px)**2 + (pixel_y - py)**2)
            distances.append((dist, i))
        
        # Sort by distance and take k nearest points
        distances.sort()
        nearest_k = distances[:k]
        
        # Apply inverse distance weighting to the k nearest points
        total_weight = 0
        weighted_lat = 0
        weighted_lon = 0
        
        for dist, idx in nearest_k:
            if dist < 1:
                dist = 1  # Avoid division by zero
            
            weight = 1 / (dist ** 2)  # Use squared distance for better weighting
            total_weight += weight
            
            # Get corresponding geo coordinates
            lat, lon = self.NBlot_geo[idx]
            weighted_lat += lat * weight
            weighted_lon += lon * weight
        
        # Compute weighted average
        if total_weight > 0:
            final_lat = weighted_lat / total_weight
            final_lon = weighted_lon / total_weight
            return final_lat, final_lon
        else:
            return None, None
    
    def compute_geolocation(self, pixel_x, pixel_y):
        """
        Legacy method that uses all points with inverse distance weighting.
        Kept for backwards compatibility.
        """
        return self.compute_geolocation_knn(pixel_x, pixel_y, k=len(self.NBlot_geo))
    
    def validate_conversion(self, pixel_x, pixel_y):
        """Validate if a pixel coordinate can be converted."""
        return self.point_in_polygon((pixel_x, pixel_y), self.NBlot_pixels)
    
    def get_conversion_confidence(self, pixel_x, pixel_y, k=None):
        """
        Get confidence score for coordinate conversion based on distance to nearest references.
        
        Returns:
            confidence (0-1): Higher values indicate more reliable conversion
        """
        if k is None:
            k = self.k
            
        if not self.NBlot_pixels or not self.point_in_polygon((pixel_x, pixel_y), self.NBlot_pixels):
            return 0.0
        
        # Calculate distance to nearest reference point
        min_distance = float('inf')
        for px, py in self.NBlot_pixels:
            dist = math.sqrt((pixel_x - px)**2 + (pixel_y - py)**2)
            min_distance = min(min_distance, dist)
        
        # Convert distance to confidence (closer = higher confidence)
        # Normalize based on image size
        max_distance = math.sqrt(self.image_width**2 + self.image_height**2)
        confidence = max(0, 1 - (min_distance / max_distance))
        
        return confidence
    
    def batch_convert(self, pixel_coordinates):
        """Convert multiple pixel coordinates to geographic coordinates efficiently."""
        results = []
        for pixel_x, pixel_y in pixel_coordinates:
            lat, lon = self.compute_geolocation_knn(pixel_x, pixel_y)
            results.append((lat, lon))
        return results
    
    def get_polygon_bounds(self):
        """Get the bounding box of the polygon in pixel coordinates."""
        if not self.NBlot_pixels:
            return None
        
        x_coords = [p[0] for p in self.NBlot_pixels]
        y_coords = [p[1] for p in self.NBlot_pixels]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
    
    def get_geo_bounds(self):
        """Get the bounding box of the reference points in geographic coordinates."""
        if not self.NBlot_geo:
            return None
        
        lats = [p[0] for p in self.NBlot_geo]
        lons = [p[1] for p in self.NBlot_geo]
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }
    
    @classmethod
    def from_files(cls, image_path, polygon_file, geo_file=None):
        """Create CoordinateConverter from files."""
        # Read YOLO polygon file
        yolo_coords = []
        if os.path.exists(polygon_file):
            with open(polygon_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                # Skip class ID, get coordinate pairs
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        x = float(parts[i])
                        y = float(parts[i+1])
                        yolo_coords.append((x, y))
        
        # Read geo coordinates if provided
        geo_coords = []
        if geo_file and os.path.exists(geo_file):
            with open(geo_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                lat = float(parts[0].strip())
                                lon = float(parts[1].strip())
                                geo_coords.append((lat, lon))
                            except ValueError:
                                continue
        
        return cls(image_path, yolo_coords, geo_coords)
    
    def debug_info(self):
        """Print debug information about the converter."""
        print("\n🔍 COORDINATE CONVERTER DEBUG INFO:")
        print(f"   Image dimensions: {self.image_width} x {self.image_height}")
        print(f"   Polygon points: {len(self.NBlot_coords)}")
        print(f"   Reference coordinates: {len(self.NBlot_geo)}")
        print(f"   K-neighbors setting: {self.k}")
        
        if self.NBlot_pixels:
            bounds = self.get_polygon_bounds()
            print(f"   Polygon bounds: {bounds}")
        
        if self.NBlot_geo:
            geo_bounds = self.get_geo_bounds()
            print(f"   Geographic bounds: {geo_bounds}")
        
        print("   Sample conversions:")
        test_points = [
            (self.image_width // 4, self.image_height // 4),
            (self.image_width // 2, self.image_height // 2),
            (3 * self.image_width // 4, 3 * self.image_height // 4)
        ]
        
        for px, py in test_points:
            lat, lon = self.compute_geolocation_knn(px, py)
            confidence = self.get_conversion_confidence(px, py)
            print(f"     ({px}, {py}) -> ({lat}, {lon}) [confidence: {confidence:.2f}]")

print("✅ ENHANCED COORDINATE CONVERTER READY!")
print("🎯 Features:")
print("  - K-nearest neighbors algorithm")
print("  - Polygon boundary validation")
print("  - Inverse distance weighting")
print("  - Confidence scoring")
print("  - Batch processing")
print("  - YOLO format support")
print("")
print("📁 Save this as: tracking/coordinate_converter.py")