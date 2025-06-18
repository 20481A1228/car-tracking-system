# tracking/streaming_detector.py - COMPLETE ENHANCED VERSION

import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time
import base64
import json
import math
from datetime import datetime
from django.core.cache import cache
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .models import Car, TrackingPoint, DetectionLog, CarTrackingSession
from .coordinate_converter import CoordinateConverter
import os

class StreamingCarDetector:
    """Enhanced streaming car detector with algorithms from standalone version."""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.session = CarTrackingSession.objects.get(id=session_id)
        
        # Initialize YOLO model
        try:
            self.model = YOLO("yolo11n.pt")
            print("✅ YOLO model loaded for enhanced streaming")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None
            
        # Initialize coordinate converter
        self.coord_converter = None
        self.init_coordinate_converter()
        
        # ENHANCED TRACKING VARIABLES (from standalone)
        self.global_car_db = {}
        self.global_car_counter = 1
        self.car_motion_history = {}  # Enhanced motion tracking
        
        # ENHANCED MOTION DETECTION SETTINGS (from standalone)
        self.motion_threshold = self.session.motion_threshold  # Default 30
        self.motion_sensitivity = self.session.motion_sensitivity  # Default 0.1
        self.min_motion_area = self.session.min_motion_area  # Default 100
        self.consecutive_motion_frames = self.session.consecutive_motion_frames  # Default 3
        self.stationary_timeout = 30  # Frames before removing cars
        
        # Streaming state
        self.is_streaming = False
        self.frames_processed = 0
        self.total_frames = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # ENHANCED MOTION DETECTION (from standalone)
        self.prev_frame = None
        
        # Video capture
        self.cap = None
        
        # WebSocket channel layer
        self.channel_layer = get_channel_layer()
        
        # Frame processing settings (from standalone)
        self.frame_skip = 3  # Process every 3rd frame like standalone
        self.cleanup_interval = 100  # Clean up every 100 frames
        
        # Debug mode
        self.debug_mode = True
        
        print(f"🎥 Enhanced StreamingCarDetector initialized for session {session_id}")
        print(f"🔧 Motion settings: threshold={self.motion_threshold}, sensitivity={self.motion_sensitivity}")

    def init_coordinate_converter(self):
        """Initialize coordinate converter from session files."""
        try:
            # Load polygon coordinates
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
            
            # Load geo coordinates with enhanced defaults (from standalone)
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
                # Use enhanced geo coordinates from standalone (34 points)
                geo_coords = [
                    (40.45844344555118, -79.92849594806206),   # Point 1
                    (40.45878807060523, -79.9286710676075),    # Point 2
                    (40.45883001768684, -79.92850243397115),   # Point 3
                    (40.458840710076004, -79.92825164548633),  # Point 4
                    (40.45889170452395, -79.92807976889544),   # Point 5
                    (40.45887607719724, -79.92789275851668),   # Point 6
                    (40.45887607719724, -79.92780736071367),   # Point 7
                    (40.458811922870545, -79.92777168821368),  # Point 8
                    (40.45879136057338, -79.92775979738035),   # Point 9
                    (40.45870253137729, -79.92780627972883),   # Point 10
                    (40.45867995650571, -79.92779287818068),   # Point 11
                    (40.45855747142186, -79.92777054134152),   # Point 12
                    (40.45842385107572, -79.92841753944151),   # Point 13
                    (40.45850174813962, -79.92836996727468),   # Point 14
                    (40.458511039609704, -79.92832279610288),  # Point 15
                    (40.45852014073843, -79.92827574799688),   # Point 16
                    (40.45852802838234, -79.92823029474195),   # Point 17
                    (40.458603871064895, -79.92822232048671),  # Point 18
                    (40.45866152436688, -79.92827730577247),   # Point 19
                    (40.45868346322101, -79.92828803460873),   # Point 20
                    (40.45863148699156, -79.92800257005571),   # Point 21
                    (40.45866061054693, -79.92793797858819),   # Point 22
                    (40.458682453205185, -79.92794834512002),  # Point 23
                    (40.45876823864577, -79.92799367114709),   # Point 24
                    (40.45878765431123, -79.92801201193417),   # Point 25
                    (40.45880949692818, -79.92802158104048),   # Point 26
                    (40.458831339538044, -79.92802796044467),  # Point 27
                    (40.458789474529596, -79.92784933712704),  # Point 28
                    (40.458811923885165, -79.9278605010844),   # Point 29
                    (40.45883194627709, -79.92787086761622),   # Point 30
                    (40.45885411567868, -79.92788266950154),   # Point 31
                    (40.45879544213203, -79.9278270136635),    # Point 32
                    (40.45881738094242, -79.92783774249975),   # Point 33
                    (40.45883931974563, -79.927848471336)      # Point 34
                ]
            
            # Initialize enhanced converter
            image_path = self.session.reference_image.path if self.session.reference_image else None
            self.coord_converter = CoordinateConverter(
                image_path=image_path,
                NBlot_coords=polygon_coords,
                geo_coords=geo_coords,
                image_width=640,
                image_height=360
            )
            self.coord_converter.k = self.session.k_neighbors
            
            print(f"✅ Enhanced coordinate converter initialized for streaming")
            print(f"📊 Polygon points: {len(polygon_coords)}, Reference points: {len(geo_coords)}")
            
        except Exception as e:
            print(f"❌ Enhanced coordinate converter error: {e}")

    def calculate_iou(self, box1, box2):
        """Enhanced IoU calculation from standalone version."""
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
        """Enhanced car ID matching from standalone version."""
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
        if self.debug_mode:
            print(f"🚗 New car detected: {new_id}")
        return new_id

    def detect_motion(self, frame):
        """Enhanced motion detection algorithm from standalone version."""
        if self.prev_frame is None:
            return None
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise (ENHANCED)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # Compute absolute difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply threshold using session settings
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise (ENHANCED)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours and create motion mask
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_mask = np.zeros_like(dilated)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_motion_area:
                cv2.drawContours(motion_mask, [cnt], 0, 255, -1)
        
        return motion_mask

    def is_moving(self, motion_mask, bbox, car_id):
        """Enhanced motion detection with consecutive frame validation from standalone."""
        if motion_mask is None:
            return False, 0.0
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = motion_mask.shape
        
        # Ensure bbox is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return False, 0.0
        
        # Extract region of interest
        roi = motion_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        
        # Calculate motion percentage
        motion_pixels = np.count_nonzero(roi)
        total_pixels = roi.size
        motion_percentage = motion_pixels / total_pixels
        
        # ENHANCED: Update motion history for this car (from standalone)
        if car_id not in self.car_motion_history:
            self.car_motion_history[car_id] = []
        
        # Store motion status (True if above threshold)
        is_moving_now = motion_percentage > self.motion_sensitivity
        self.car_motion_history[car_id].append(is_moving_now)
        
        # Keep only recent history
        if len(self.car_motion_history[car_id]) > self.consecutive_motion_frames:
            self.car_motion_history[car_id] = self.car_motion_history[car_id][-self.consecutive_motion_frames:]
        
        # ENHANCED: Car is considered moving if it has been moving for consecutive frames
        if len(self.car_motion_history[car_id]) >= self.consecutive_motion_frames:
            recent_motion = self.car_motion_history[car_id][-self.consecutive_motion_frames:]
            consistently_moving = sum(recent_motion) >= (self.consecutive_motion_frames * 0.7)  # 70% of frames
            return consistently_moving, motion_percentage
        
        # Not enough history, use current frame only
        return is_moving_now, motion_percentage

    def pixel_to_geo(self, x, y):
        """Enhanced coordinate conversion using K-NN from standalone."""
        if self.coord_converter is None:
            return 0, 0
        
        try:
            # Use enhanced K-NN method with polygon validation
            lat, lon = self.coord_converter.compute_geolocation_knn(x, y)
            if lat is None or lon is None:
                return 0, 0
            return lat, lon
        except Exception as e:
            return 0, 0

    def calculate_bbox_center(self, bbox):
        """Calculate center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def cleanup_stationary_cars(self):
        """Enhanced cleanup from standalone version."""
        cars_to_remove = []
        current_frame = self.frames_processed
        
        for car_id, car_data in self.global_car_db.items():
            last_seen = car_data.get('last_seen', 0)
            frames_since_seen = current_frame - last_seen
            
            # Remove cars not seen recently
            if frames_since_seen > self.stationary_timeout:
                cars_to_remove.append(car_id)
        
        # Remove cars from tracking
        for car_id in cars_to_remove:
            if car_id in self.global_car_db:
                del self.global_car_db[car_id]
            if car_id in self.car_motion_history:
                del self.car_motion_history[car_id]
            if self.debug_mode:
                print(f"🗑️ Removed stationary car: {car_id}")

    def draw_detections(self, frame, detections):
        """Enhanced drawing with standalone styling."""
        overlay_frame = frame.copy()
        moving_cars_count = len(detections)
        
        # Draw enhanced title and info (like standalone)
        cv2.putText(overlay_frame, f"Enhanced Moving Cars with Improved Coordinates | {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay_frame, f"Moving Cars: {moving_cars_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw each car detection with enhanced styling
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            car_id = detection['car_id']
            lat, lon = detection['lat'], detection['lon']
            
            # Draw bright green bounding box (moving cars only)
            cv2.rectangle(overlay_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            
            # Enhanced label with coordinates and motion confidence
            coord_display = f"({lat:.6f}, {lon:.6f})"
            main_label = f"{car_id} {coord_display}"
            confidence_text = f"Motion: {detection.get('motion_confidence', 0):.1%}"
            
            # Draw enhanced labels with outline for visibility
            label_y = int(y1) - 15
            
            # Main label with outline
            cv2.putText(overlay_frame, main_label, (int(x1) + 5, label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Black outline
            cv2.putText(overlay_frame, main_label, (int(x1) + 5, label_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Cyan text
            
            # Motion confidence label
            cv2.putText(overlay_frame, confidence_text, (int(x1) + 5, label_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
            cv2.putText(overlay_frame, confidence_text, (int(x1) + 5, label_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # Yellow text
            
            # Draw center point
            center_x, center_y = self.calculate_bbox_center((x1, y1, x2, y2))
            cv2.circle(overlay_frame, (int(center_x), int(center_y)), 3, (0, 255, 0), -1)
        
        # Enhanced bottom info panel (like standalone)
        cv2.putText(overlay_frame, f"ENHANCED MOVING CARS + IMPROVED COORDINATES — Active: {moving_cars_count}", 
                   (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(overlay_frame, f"Motion Detection: Threshold={self.motion_threshold} | MinArea={self.min_motion_area} | Sensitivity={self.motion_sensitivity*100:.1f}%", 
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(overlay_frame, f"Coordinate System: K-NN with polygon boundary checking | K={self.coord_converter.k if self.coord_converter else 'N/A'}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(overlay_frame, "Green Boxes = Moving Cars ONLY | Coordinates with polygon validation", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame

    def process_frame(self, frame):
        """Enhanced frame processing with standalone algorithms."""
        if self.model is None:
            return frame, []
        
        frame = cv2.resize(frame, (640, 360))
        timestamp = datetime.now()
        
        # Periodic cleanup (from standalone)
        if self.frames_processed % self.cleanup_interval == 0:
            self.cleanup_stationary_cars()
        
        # Enhanced motion detection
        motion_mask = self.detect_motion(frame)
        self.prev_frame = frame.copy()
        
        detections = []
        total_yolo_detections = 0
        moving_detections = 0
        
        try:
            # YOLO detection with enhanced settings
            results = self.model(frame, classes=[2], conf=0.3, imgsz=1088)
            
            for result in results:
                boxes = result.boxes.data
                total_yolo_detections = len(boxes)
                
                for box in boxes:
                    if len(box) < 5:
                        continue
                    
                    x1, y1, x2, y2, conf = box[:5].tolist()
                    
                    # Enhanced car ID matching
                    car_id = self.match_car_id((x1, y1, x2, y2))
                    
                    # Enhanced motion detection
                    car_is_moving, motion_confidence = self.is_moving(motion_mask, (x1, y1, x2, y2), car_id)
                    
                    # DEBUG: Log motion detection results
                    if self.debug_mode and total_yolo_detections > 0:
                        print(f"🔍 Car {car_id}: moving={car_is_moving}, confidence={motion_confidence:.2f}")
                    
                    # CRITICAL: Only show moving cars
                    if not car_is_moving:
                        continue
                    
                    moving_detections += 1
                    
                    # Enhanced coordinate conversion
                    center_x, center_y = self.calculate_bbox_center((x1, y1, x2, y2))
                    lat, lon = self.pixel_to_geo(center_x, center_y)
                    
                    # Skip if outside polygon
                    if lat == 0 and lon == 0:
                        if self.debug_mode:
                            print(f"⚠️ Car {car_id} outside polygon: ({center_x:.1f}, {center_y:.1f})")
                        continue
                    
                    # FIXED: Create properly formatted detection
                    detection = {
                        'car_id': car_id,
                        'bbox': (x1, y1, x2, y2),
                        'lat': lat,
                        'lon': lon,
                        'motion_confidence': motion_confidence,
                        'center': (center_x, center_y),
                        'timestamp': timestamp.isoformat()
                    }
                    detections.append(detection)
                    
                    # DEBUG: Log successful detection
                    if self.debug_mode:
                        print(f"✅ MOVING CAR: {car_id} at ({lat:.6f}, {lon:.6f})")
        
        except Exception as e:
            print(f"❌ Enhanced detection error: {e}")
            import traceback
            traceback.print_exc()
        
        # DEBUG: Log detection summary
        if self.debug_mode:
            print(f"📊 Frame {self.frames_processed}: YOLO={total_yolo_detections}, Moving={moving_detections}, Final={len(detections)}")
        
        # Enhanced frame drawing
        processed_frame = self.draw_detections(frame, detections)
        
        return processed_frame, detections

    def frame_to_base64(self, frame):
        """Convert frame to base64 for web transmission."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64

    def broadcast_frame(self, frame_base64, detections):
        """Enhanced frame broadcasting with proper detection data."""
        if self.channel_layer:
            try:
                # Calculate FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # FIXED: Ensure detections data is properly formatted
                formatted_detections = []
                for d in detections:
                    detection_data = {
                        'car_id': str(d['car_id']),
                        'lat': float(d['lat']),
                        'lon': float(d['lon']),
                        'coordinates': f"({d['lat']:.6f}, {d['lon']:.6f})",
                        'motion_confidence': float(d.get('motion_confidence', 0))
                    }
                    formatted_detections.append(detection_data)
                
                # FIXED: Enhanced data for WebSocket
                stream_data = {
                    'frame': frame_base64,
                    'moving_cars': len(detections),  # This should match detection count
                    'total_cars': len(self.global_car_db),
                    'fps': self.current_fps,
                    'progress': 0,
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'detections': formatted_detections,  # CRITICAL: Properly formatted detections
                    'motion_settings': {
                        'threshold': self.motion_threshold,
                        'sensitivity': self.motion_sensitivity,
                        'min_area': self.min_motion_area,
                        'consecutive_frames': self.consecutive_motion_frames
                    },
                    'debug_info': {
                        'total_detections': len(detections),
                        'global_cars': len(self.global_car_db),
                        'frame_processed': self.frames_processed
                    } if self.debug_mode else {}
                }
                
                # DEBUG: Log what we're sending
                if self.debug_mode and len(detections) > 0:
                    print(f"🔄 Broadcasting {len(detections)} detections via WebSocket:")
                    for det in formatted_detections:
                        print(f"   {det['car_id']}: {det['coordinates']}")
                
                # Send via WebSocket
                async_to_sync(self.channel_layer.group_send)(
                    f'streaming_{self.session_id}',
                    {
                        'type': 'stream_frame',
                        **stream_data
                    }
                )
                
                # DEBUG: Confirm broadcast
                if self.debug_mode:
                    print(f"✅ WebSocket broadcast sent: {len(detections)} cars, FPS: {self.current_fps}")
                
            except Exception as e:
                print(f"❌ Enhanced WebSocket broadcast error: {e}")
                import traceback
                traceback.print_exc()

    def start_streaming(self):
        """Start enhanced live video streaming."""
        if self.is_streaming:
            return False, "Already streaming"
        
        if not self.session.video_file or not os.path.exists(self.session.video_file.path):
            return False, "Video file not found"
        
        self.cap = cv2.VideoCapture(self.session.video_file.path)
        if not self.cap.isOpened():
            return False, "Could not open video file"
        
        self.is_streaming = True
        self.frames_processed = 0
        
        print(f"🎥 Starting enhanced live streaming for session {self.session_id}")
        print(f"🔧 Enhanced motion settings: threshold={self.motion_threshold}, sensitivity={self.motion_sensitivity}")
        
        # Start enhanced streaming thread
        streaming_thread = threading.Thread(target=self._enhanced_streaming_loop)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        return True, "Enhanced streaming started"

    def _enhanced_streaming_loop(self):
        """Enhanced streaming loop with standalone algorithms."""
        try:
            frame_count = 0
            
            while self.is_streaming and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video for continuous streaming
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0  # Reset frame count when looping
                    continue
                
                frame_count += 1
                self.frames_processed += 1
                
                # Enhanced frame skipping (from standalone)
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Enhanced frame processing
                processed_frame, detections = self.process_frame(frame)
                
                # Convert to base64
                frame_base64 = self.frame_to_base64(processed_frame)
                
                # Enhanced broadcasting
                self.broadcast_frame(frame_base64, detections)
                
                # Control streaming rate (approximately 10 FPS)
                time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Enhanced streaming loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_streaming()

    def stop_streaming(self):
        """Stop enhanced live video streaming."""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f"🛑 Stopped enhanced streaming for session {self.session_id}")

print("✅ ENHANCED STREAMING DETECTOR WITH STANDALONE ALGORITHMS")
print("🚗 Features from standalone version:")
print("  - Enhanced motion detection with consecutive frame validation")  
print("  - Better car ID matching with improved IoU")
print("  - Motion history tracking for consistent detection")
print("  - Enhanced coordinate conversion with K-NN")
print("  - Better cleanup and stationary car removal")
print("  - Improved visual styling and debugging")
print("  - Fixed WebSocket data transmission")
print("")
print("📁 Replace your tracking/streaming_detector.py with this complete version!")