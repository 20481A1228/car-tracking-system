import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
from datetime import datetime
from django.core.cache import cache
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Car, TrackingPoint, DetectionLog, CarTrackingSession
from .coordinate_converter import CoordinateConverter
import math
import json
import os

class CarDetector:
    """Enhanced Car Detector for Django integration with streaming capabilities."""
    
    def __init__(self, session_id):
        self.session = CarTrackingSession.objects.get(id=session_id)
        
        try:
            self.model = YOLO("yolo11n.pt")
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None
            
        # Initialize coordinate converter
        self.coord_converter = None
        self.init_coordinate_converter()
        
        # Tracking variables
        self.global_car_db = {}
        self.global_car_counter = 1
        self.car_motion_history = {}
        
        # Processing state
        self.frames_processed = 0
        self.total_frames = 0
        self.processing_active = False
        
        # Motion detection
        self.prev_frame = None
        
        # WebSocket channel layer for real-time updates
        self.channel_layer = get_channel_layer()

    def init_coordinate_converter(self):
        """Initialize coordinate converter from session files."""
        try:
            # Load polygon coordinates from YOLO format file
            polygon_coords = []
            if self.session.polygon_file and os.path.exists(self.session.polygon_file.path):
                with open(self.session.polygon_file.path, 'r') as f:
                    line = f.readline().strip()
                    parts = line.split()
                    for i in range(1, len(parts), 2):
                        if i+1 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i+1])
                            polygon_coords.append((x, y))
            
            # Load geo coordinates if available
            geo_coords = []
            if self.session.geo_coords_file and os.path.exists(self.session.geo_coords_file.path):
                with open(self.session.geo_coords_file.path, 'r') as f:
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
            else:
                # Use reference points from session or default coordinates
                reference_points = self.session.reference_points.all()
                if reference_points.exists():
                    geo_coords = [(rp.latitude, rp.longitude) for rp in reference_points]
                else:
                    # Default coordinates (like in your standalone app)
                    geo_coords = [
                        (40.45844344555118, -79.92849594806206),
                        (40.45878807060523, -79.9286710676075),
                        (40.45883001768684, -79.92850243397115),
                        (40.458840710076004, -79.92825164548633),
                        (40.45889170452395, -79.92807976889544),
                        (40.45887607719724, -79.92789275851668),
                        (40.45887607719724, -79.92780736071367),
                        (40.458811922870545, -79.92777168821368),
                        (40.45879136057338, -79.92775979738035),
                        (40.45870253137729, -79.92780627972883),
                        (40.45867995650571, -79.92779287818068),
                        (40.45855747142186, -79.92777054134152),
                        (40.45842385107572, -79.92841753944151),
                        (40.45850174813962, -79.92836996727468),
                        (40.458511039609704, -79.92832279610288),
                        (40.45852014073843, -79.92827574799688),
                        (40.45852802838234, -79.92823029474195),
                        (40.458603871064895, -79.92822232048671),
                        (40.45866152436688, -79.92827730577247),
                        (40.45868346322101, -79.92828803460873),
                        (40.45863148699156, -79.92800257005571),
                        (40.45866061054693, -79.92793797858819),
                        (40.458682453205185, -79.92794834512002),
                        (40.45876823864577, -79.92799367114709),
                        (40.45878765431123, -79.92801201193417),
                        (40.45880949692818, -79.92802158104048),
                        (40.458831339538044, -79.92802796044467),
                        (40.458789474529596, -79.92784933712704),
                        (40.458811923885165, -79.9278605010844),
                        (40.45883194627709, -79.92787086761622),
                        (40.45885411567868, -79.92788266950154),
                        (40.45879544213203, -79.9278270136635),
                        (40.45881738094242, -79.92783774249975),
                        (40.45883931974563, -79.927848471336)
                    ]
            
            # Initialize converter
            image_path = self.session.reference_image.path if self.session.reference_image else None
            self.coord_converter = CoordinateConverter(
                image_path=image_path,
                NBlot_coords=polygon_coords,
                geo_coords=geo_coords,
                image_width=640,
                image_height=360
            )
            self.coord_converter.k = self.session.k_neighbors
            
            self.log_message(f"Coordinate converter initialized with {len(polygon_coords)} polygon points and {len(geo_coords)} reference points")
            
        except Exception as e:
            self.log_message(f"Failed to initialize coordinate converter: {str(e)}", "ERROR")
            print(f"❌ Coordinate converter error: {e}")

    def log_message(self, message, level="INFO"):
        """Log message to database and console."""
        try:
            DetectionLog.objects.create(
                session=self.session,
                message=message,
                level=level
            )
        except Exception as e:
            print(f"Logging error: {e}")
        print(f"[{level}] {message}")

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def match_car_id(self, bbox, iou_thresh=0.5):
        """Assign consistent IDs using IoU matching."""
        best_match = None
        best_iou = 0
        
        for car_id, saved_data in self.global_car_db.items():
            iou = self.calculate_iou(bbox, saved_data['bbox'])
            if iou > iou_thresh and iou > best_iou:
                best_iou = iou
                best_match = car_id
        
        if best_match is not None:
            self.global_car_db[best_match]['bbox'] = bbox
            self.global_car_db[best_match]['last_seen'] = self.frames_processed
            return best_match
        
        # Create new car
        new_id = f"car-{self.global_car_counter}"
        self.global_car_db[new_id] = {
            'bbox': bbox, 
            'last_seen': self.frames_processed
        }
        self.global_car_counter += 1
        print(f"🚗 New car detected: {new_id}")
        return new_id

    def detect_motion(self, frame):
        """Detect motion between frames using enhanced algorithm."""
        if self.prev_frame is None:
            return None
        
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_diff, self.session.motion_threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_mask = np.zeros_like(dilated)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.session.min_motion_area:
                cv2.drawContours(motion_mask, [cnt], 0, 255, -1)
        
        return motion_mask

    def is_moving(self, motion_mask, bbox, car_id):
        """Determine if a car is moving using enhanced motion detection."""
        if motion_mask is None:
            return False, 0.0
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = motion_mask.shape
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return False, 0.0
        
        roi = motion_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        
        motion_pixels = np.count_nonzero(roi)
        total_pixels = roi.size
        motion_percentage = motion_pixels / total_pixels
        
        if car_id not in self.car_motion_history:
            self.car_motion_history[car_id] = []
        
        is_moving_now = motion_percentage > self.session.motion_sensitivity
        self.car_motion_history[car_id].append(is_moving_now)
        
        if len(self.car_motion_history[car_id]) > self.session.consecutive_motion_frames:
            self.car_motion_history[car_id] = self.car_motion_history[car_id][-self.session.consecutive_motion_frames:]
        
        if len(self.car_motion_history[car_id]) >= self.session.consecutive_motion_frames:
            recent_motion = self.car_motion_history[car_id][-self.session.consecutive_motion_frames:]
            consistently_moving = sum(recent_motion) >= (self.session.consecutive_motion_frames * 0.7)
            return consistently_moving, motion_percentage
        
        return is_moving_now, motion_percentage

    def pixel_to_geo(self, x, y):
        """Convert pixel coordinates to geographic coordinates."""
        if self.coord_converter is None:
            return 0, 0
        
        try:
            lat, lon = self.coord_converter.compute_geolocation_knn(x, y)
            if lat is None or lon is None:
                return 0, 0
            return lat, lon
        except Exception as e:
            print(f"Coordinate conversion error: {e}")
            return 0, 0

    def calculate_bbox_center(self, bbox):
        """Calculate center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def calculate_speed(self, prev_geo_pos, current_geo_pos, time_diff_seconds):
        """Calculate speed in km/h based on geographic distance and time difference."""
        if prev_geo_pos is None or time_diff_seconds <= 0:
            return 0.0
        
        lat1, lon1 = prev_geo_pos
        lat2, lon2 = current_geo_pos
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        
        distance = c * r
        speed = (distance / time_diff_seconds) * 3.6  # Convert to km/h
        
        return speed

    def cleanup_stationary_cars(self):
        """Remove cars that haven't been seen for a while."""
        cars_to_remove = []
        current_frame = self.frames_processed
        stationary_timeout = 30  # frames
        
        for car_id, car_data in self.global_car_db.items():
            last_seen = car_data.get('last_seen', 0)
            frames_since_seen = current_frame - last_seen
            
            if frames_since_seen > stationary_timeout:
                cars_to_remove.append(car_id)
        
        for car_id in cars_to_remove:
            if car_id in self.global_car_db:
                del self.global_car_db[car_id]
            if car_id in self.car_motion_history:
                del self.car_motion_history[car_id]
            print(f"🗑️ Removed stationary car: {car_id}")

    def broadcast_live_data(self, cars_data):
        """Broadcast live data via WebSocket if available."""
        if self.channel_layer:
            try:
                async_to_sync(self.channel_layer.group_send)(
                    f'tracking_{self.session.id}',
                    {
                        'type': 'tracking_update',
                        'data': {
                            'cars': cars_data,
                            'processed_frames': self.frames_processed,
                            'total_frames': self.total_frames,
                            'session_status': self.session.status
                        }
                    }
                )
            except Exception as e:
                print(f"WebSocket broadcast error: {e}")

    def process_video(self):
        """Main video processing function with enhanced car detection."""
        self.processing_active = True
        self.session.status = 'processing'
        self.session.save()
        
        if self.model is None:
            self.log_message("YOLO model not available", "ERROR")
            self.session.status = 'failed'
            self.session.save()
            return
        
        try:
            # Check if video file exists
            if not self.session.video_file or not os.path.exists(self.session.video_file.path):
                raise Exception("Video file not found")
            
            cap = cv2.VideoCapture(self.session.video_file.path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.session.total_frames = self.total_frames
            self.session.save()
            
            self.log_message(f"Starting video processing. Total frames: {self.total_frames}, FPS: {fps}")
            
            frame_skip = 3  # Process every 3rd frame for performance
            cars_database = {}
            moving_cars_count = 0
            cleanup_interval = 100
            
            while cap.isOpened() and self.processing_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frames_processed += 1
                
                if self.frames_processed % frame_skip != 0:
                    continue
                
                frame = cv2.resize(frame, (640, 360))
                timestamp = datetime.now()
                
                # Periodic cleanup
                if self.frames_processed % cleanup_interval == 0:
                    self.cleanup_stationary_cars()
                
                # Motion detection
                motion_mask = self.detect_motion(frame)
                self.prev_frame = frame.copy()
                
                moving_cars_count = 0
                current_cars_data = []
                
                # YOLO detection
                try:
                    results = self.model(frame, classes=[2], conf=0.1, imgsz=640, iou=0.4)
                    
                    for result in results:
                        boxes = result.boxes.data
                        for box in boxes:
                            if len(box) < 5:
                                continue
                            
                            x1, y1, x2, y2, conf = box[:5].tolist()
                            
                            # Match car ID
                            car_id = self.match_car_id((x1, y1, x2, y2))
                            
                            # Check if moving
                            car_is_moving, motion_confidence = self.is_moving(motion_mask, (x1, y1, x2, y2), car_id)
                            
                            # Only process moving cars
                            if not car_is_moving:
                                continue
                            
                            moving_cars_count += 1
                            
                            # Calculate center and convert to geo coordinates
                            center_x, center_y = self.calculate_bbox_center((x1, y1, x2, y2))
                            lat, lon = self.pixel_to_geo(center_x, center_y)
                            
                            if lat == 0 and lon == 0:
                                print(f"⚠️ Car at pixel ({center_x:.1f}, {center_y:.1f}) is outside polygon")
                                continue
                            
                            # Get or create car in database
                            car, created = Car.objects.get_or_create(
                                session=self.session,
                                car_id=car_id,
                                defaults={
                                    'total_detections': 0,
                                    'max_speed': 0.0,
                                    'avg_speed': 0.0,
                                    'is_active': True
                                }
                            )
                            
                            # Calculate speed
                            speed = 0.0
                            if car_id in cars_database:
                                prev_data = cars_database[car_id]
                                time_diff = (timestamp - prev_data['timestamp']).total_seconds()
                                if time_diff > 0:
                                    speed = self.calculate_speed(
                                        (prev_data['lat'], prev_data['lon']),
                                        (lat, lon),
                                        time_diff
                                    )
                            
                            # Store tracking point
                            TrackingPoint.objects.create(
                                car=car,
                                timestamp=timestamp,
                                pixel_x=center_x,
                                pixel_y=center_y,
                                latitude=lat,
                                longitude=lon,
                                is_moving=True,
                                speed=speed,
                                motion_confidence=motion_confidence,
                                bbox_x1=x1,
                                bbox_y1=y1,
                                bbox_x2=x2,
                                bbox_y2=y2
                            )
                            
                            # Update car stats
                            car.total_detections += 1
                            car.max_speed = max(car.max_speed, speed)
                            car.last_seen = timestamp
                            car.save()
                            
                            # Store for next iteration
                            cars_database[car_id] = {
                                'lat': lat,
                                'lon': lon,
                                'timestamp': timestamp
                            }
                            
                            # Prepare data for live broadcast
                            current_cars_data.append({
                                'car_id': car_id,
                                'latitude': lat,
                                'longitude': lon,
                                'speed': speed,
                                'coordinates': f"({lat:.6f}, {lon:.6f})",
                                'motion_confidence': motion_confidence
                            })
                            
                            print(f"🚗 MOVING CAR: {car_id} ({lat:.6f}, {lon:.6f}) Speed: {speed:.1f}km/h")
                
                except Exception as e:
                    self.log_message(f"YOLO processing error: {str(e)}", "WARNING")
                
                # Broadcast live data
                self.broadcast_live_data(current_cars_data)
                
                # Update progress
                self.session.processed_frames = self.frames_processed
                if self.frames_processed % 50 == 0:  # Save every 50 frames
                    self.session.save()
                
                if self.frames_processed % 100 == 0:
                    self.log_message(f"Processed {self.frames_processed}/{self.total_frames} frames. Moving cars: {moving_cars_count}")
            
            cap.release()
            
            # Update final stats
            total_cars_detected = Car.objects.filter(session=self.session).count()
            self.session.status = 'completed'
            self.session.cars_detected = total_cars_detected
            self.session.processed_frames = self.frames_processed
            self.session.save()
            
            self.log_message(f"✅ Video processing completed! Detected {total_cars_detected} cars across {self.frames_processed} frames.")
            
        except Exception as e:
            self.session.status = 'failed'
            self.session.save()
            self.log_message(f"❌ Video processing failed: {str(e)}", "ERROR")
            print(f"❌ Processing error: {e}")
        
        finally:
            self.processing_active = False

print("✅ ENHANCED CAR DETECTOR READY!")
print("🚗 Features:")
print("  - YOLO-based car detection")
print("  - Enhanced motion detection")
print("  - Coordinate conversion with polygon validation")
print("  - Real-time car tracking and ID assignment")
print("  - Speed calculation")
print("  - WebSocket broadcasting")
print("  - Database integration")
print("")
print("📁 Save this as: tracking/car_detector.py")