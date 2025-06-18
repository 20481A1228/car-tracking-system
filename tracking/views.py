# tracking/views.py - COMPLETE VIEWS WITH STREAMING FUNCTIONALITY

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.cache import cache
from django.db.models import Count, Avg, Max, Sum, Min
import json
import threading
import math
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from .models import CarTrackingSession, Car, TrackingPoint, DetectionLog
from .forms import CarTrackingSessionForm

# Try to import CarDetector and StreamingCarDetector
try:
    from .car_detector import CarDetector
except ImportError:
    CarDetector = None
    print("Warning: CarDetector not found. Background processing will be limited.")

try:
    from .streaming_detector import StreamingCarDetector
except ImportError:
    StreamingCarDetector = None
    print("Warning: StreamingCarDetector not found. Live streaming will be limited.")

# Global streaming detectors (for live streaming)
active_streams = {}

# ==================================================
# MAIN DASHBOARD AND SESSION VIEWS
# ==================================================

def dashboard(request):
    """Dashboard view with fixed statistics calculation."""
    sessions = CarTrackingSession.objects.all().order_by('-created_at')
    
    # Calculate statistics properly
    total_sessions = sessions.count()
    active_processing = sessions.filter(status='processing').count()
    completed_sessions = sessions.filter(status='completed').count()
    failed_sessions = sessions.filter(status='failed').count()
    
    # Sum all cars detected across sessions
    total_cars_detected = sessions.aggregate(
        total=Sum('cars_detected')
    )['total'] or 0
    
    context = {
        'sessions': sessions,
        'total_sessions': total_sessions,
        'active_processing': active_processing,
        'completed_sessions': completed_sessions,
        'failed_sessions': failed_sessions,
        'total_cars_detected': total_cars_detected,
    }
    return render(request, 'dashboard.html', context)

def session_detail(request, session_id):
    """Session detail view with proper relationship handling."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    cars = session.cars.all()
    
    # Get detection logs with proper error handling
    try:
        logs = session.detection_logs.order_by('-timestamp')[:20]
    except AttributeError:
        logs = DetectionLog.objects.filter(session=session).order_by('-timestamp')[:20]
    
    # Calculate additional statistics
    active_cars = cars.filter(is_active=True).count()
    total_detections = cars.aggregate(total=Sum('total_detections'))['total'] or 0
    
    context = {
        'session': session,
        'cars': cars,
        'logs': logs,
        'active_cars': active_cars,
        'total_detections': total_detections,
    }
    return render(request, 'session_detail.html', context)

def create_session(request):
    """Create new tracking session with improved error handling."""
    if request.method == 'POST':
        form = CarTrackingSessionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                session = form.save()
                print(f"✅ Session created: {session.name} (ID: {session.id})")
                messages.success(request, f'Session "{session.name}" created successfully!')
                return redirect('session_detail', session_id=session.id)
            except Exception as e:
                print(f"❌ Error creating session: {str(e)}")
                messages.error(request, f'Error creating session: {str(e)}')
        else:
            print("❌ Form validation errors:", form.errors)
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CarTrackingSessionForm()
    
    return render(request, 'create_session.html', {'form': form})

def start_processing(request, session_id):
    """Start video processing for a session."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    if session.status == 'processing':
        messages.warning(request, 'Session is already being processed.')
        return redirect('session_detail', session_id=session.id)
    
    if session.status == 'completed':
        messages.info(request, 'Session has already been completed.')
        return redirect('session_detail', session_id=session.id)
    
    # Start processing in background thread
    try:
        if CarDetector is None:
            messages.error(request, 'Car detection system is not properly configured.')
            return redirect('session_detail', session_id=session.id)
            
        detector = CarDetector(session_id)
        thread = threading.Thread(target=detector.process_video)
        thread.daemon = True
        thread.start()
        
        session.status = 'processing'
        session.save()
        
        messages.success(request, 'Video processing started!')
    except Exception as e:
        session.status = 'failed'
        session.save()
        messages.error(request, f'Failed to start processing: {str(e)}')
    
    return redirect('session_detail', session_id=session.id)

def visualization(request, session_id):
    """Live visualization view with car tracking data."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    cars = session.cars.filter(is_active=True)
    
    # Get all tracking points for visualization
    all_points = []
    total_cars = 0
    
    for car in cars:
        car_points = []
        tracking_points = car.tracking_points.all().order_by('timestamp')
        
        for point in tracking_points:
            car_points.append({
                'lat': float(point.latitude),
                'lng': float(point.longitude),
                'timestamp': point.timestamp.isoformat(),
                'speed': float(point.speed),
                'car_id': car.car_id,
                'motion_confidence': float(point.motion_confidence)
            })
        
        if car_points:
            all_points.append({
                'car_id': car.car_id,
                'points': car_points,
                'total_points': len(car_points),
                'max_speed': float(car.max_speed),
                'avg_speed': float(car.avg_speed),
                'total_detections': car.total_detections
            })
            total_cars += 1
    
    context = {
        'session': session,
        'cars_data': json.dumps(all_points),
        'total_cars': total_cars,
        'cars': cars
    }
    return render(request, 'visualization.html', context)

def car_detail(request, car_id):
    """Car detail view with path analysis."""
    car = get_object_or_404(Car, id=car_id)
    tracking_points = car.tracking_points.all().order_by('timestamp')
    
    # Calculate path statistics using Haversine formula
    total_distance = 0.0
    path_data = []
    
    # Convert tracking points to path data for JavaScript
    for point in tracking_points:
        path_data.append({
            'lat': float(point.latitude),
            'lng': float(point.longitude),
            'timestamp': point.timestamp.isoformat(),
            'speed': float(point.speed),
            'motion_confidence': float(point.motion_confidence)
        })
    
    # Calculate total distance if we have multiple points
    if len(tracking_points) > 1:
        for i in range(1, len(tracking_points)):
            prev = tracking_points[i-1]
            curr = tracking_points[i]
            
            # Haversine formula for distance calculation
            lat1, lon1 = math.radians(float(prev.latitude)), math.radians(float(prev.longitude))
            lat2, lon2 = math.radians(float(curr.latitude)), math.radians(float(curr.longitude))
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            distance = c * 6371000  # Earth radius in meters
            total_distance += distance
    
    # Calculate additional statistics
    avg_speed = tracking_points.aggregate(avg_speed=Avg('speed'))['avg_speed'] or 0
    max_speed = tracking_points.aggregate(max_speed=Max('speed'))['max_speed'] or 0
    
    # Calculate time-based statistics
    time_span = None
    if tracking_points.exists():
        first_point = tracking_points.first()
        last_point = tracking_points.last()
        if first_point and last_point:
            time_span = (last_point.timestamp - first_point.timestamp).total_seconds()
    
    context = {
        'car': car,
        'tracking_points': tracking_points,
        'total_distance': total_distance,
        'path_data': json.dumps(path_data),
        'avg_speed_calculated': avg_speed,
        'max_speed_calculated': max_speed,
        'time_span': time_span,
        'total_points': tracking_points.count()
    }
    return render(request, 'car_detail.html', context)

@csrf_exempt
@require_http_methods(["GET"])
def live_data(request, session_id):
    """API endpoint for live car tracking data."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    cars = session.cars.filter(is_active=True)
    
    live_data = []
    for car in cars:
        latest_point = car.tracking_points.last()
        if latest_point:
            live_data.append({
                'car_id': car.car_id,
                'latitude': float(latest_point.latitude),
                'longitude': float(latest_point.longitude),
                'speed': float(latest_point.speed),
                'timestamp': latest_point.timestamp.isoformat(),
                'motion_confidence': float(latest_point.motion_confidence),
                'is_moving': latest_point.is_moving
            })
    
    # Calculate progress percentage properly
    progress_percentage = 0.0
    if session.total_frames > 0:
        progress_percentage = (session.processed_frames / session.total_frames) * 100
    
    return JsonResponse({
        'success': True,
        'cars': live_data,
        'session_status': session.status,
        'processed_frames': session.processed_frames,
        'total_frames': session.total_frames,
        'progress_percentage': round(progress_percentage, 2),
        'cars_detected': session.cars_detected,
        'active_cars': len(live_data),
        'timestamp': session.updated_at.isoformat() if session.updated_at else None
    })

def analytics(request):
    """Analytics dashboard with comprehensive statistics."""
    # Basic counts
    total_sessions = CarTrackingSession.objects.count()
    total_cars = Car.objects.count()
    total_tracking_points = TrackingPoint.objects.count()
    
    # Session statistics
    session_stats = CarTrackingSession.objects.aggregate(
        avg_cars=Avg('cars_detected'),
        max_cars=Max('cars_detected'),
        total_cars_detected=Sum('cars_detected')
    )
    
    # Speed statistics across all tracking points
    speed_stats = TrackingPoint.objects.aggregate(
        max_speed=Max('speed'),
        avg_speed=Avg('speed'),
        min_speed=Min('speed')
    )
    
    # Motion confidence statistics
    confidence_stats = TrackingPoint.objects.aggregate(
        avg_confidence=Avg('motion_confidence'),
        max_confidence=Max('motion_confidence')
    )
    
    # Status distribution
    status_distribution = CarTrackingSession.objects.values('status').annotate(
        count=Count('id')
    ).order_by('status')
    
    # Recent activity
    recent_sessions = CarTrackingSession.objects.order_by('-created_at')[:10]
    recent_cars = Car.objects.order_by('-first_detected')[:10]
    
    # Time-based statistics
    today = timezone.now().date()
    week_ago = today - timedelta(days=7)
    
    recent_sessions_count = CarTrackingSession.objects.filter(
        created_at__date__gte=week_ago
    ).count()
    
    stats = {
        'total_sessions': total_sessions,
        'total_cars': total_cars,
        'total_tracking_points': total_tracking_points,
        'avg_cars_per_session': round(session_stats['avg_cars'] or 0, 1),
        'max_cars_per_session': session_stats['max_cars'] or 0,
        'total_cars_detected': session_stats['total_cars_detected'] or 0,
        'max_speed_recorded': round(speed_stats['max_speed'] or 0, 1),
        'avg_speed': round(speed_stats['avg_speed'] or 0, 1),
        'min_speed': round(speed_stats['min_speed'] or 0, 1),
        'avg_motion_confidence': round(confidence_stats['avg_confidence'] or 0, 2),
        'max_motion_confidence': round(confidence_stats['max_confidence'] or 0, 2),
        'recent_sessions': recent_sessions,
        'recent_cars': recent_cars,
        'status_distribution': status_distribution,
        'recent_sessions_count': recent_sessions_count
    }
    
    return render(request, 'analytics.html', {'stats': stats})

# ==================================================
# LIVE STREAMING VIEWS (ENHANCED)
# ==================================================

def live_stream_view(request, session_id):
    """View for live video streaming page."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    context = {
        'session': session,
        'streaming_available': StreamingCarDetector is not None,
    }
    return render(request, 'live_stream.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def start_video_streaming(request, session_id):
    """Start real-time video streaming with car detection."""
    global active_streams
    
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    # Check if streaming is available
    if StreamingCarDetector is None:
        return JsonResponse({
            'success': False, 
            'error': 'Streaming detector not available. Please create streaming_detector.py first.'
        })
    
    # Check if already streaming
    if session_id in active_streams:
        return JsonResponse({
            'success': False,
            'error': 'Stream already active for this session'
        })
    
    try:
        # Create enhanced streaming detector
        detector = StreamingCarDetector(session_id)
        detector.debug_mode = True  # Enable debugging
        
        # Start streaming
        success, message = detector.start_streaming()
        
        if success:
            active_streams[session_id] = detector
            
            return JsonResponse({
                'success': True,
                'message': 'Enhanced live streaming started successfully!',
                'session_id': session_id
            })
        else:
            return JsonResponse({
                'success': False,
                'error': message
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to start streaming: {str(e)}'
        })

@csrf_exempt  
@require_http_methods(["POST"])
def stop_video_streaming(request, session_id):
    """Stop real-time video streaming."""
    global active_streams
    
    if session_id in active_streams:
        try:
            detector = active_streams[session_id]
            detector.stop_streaming()
            del active_streams[session_id]
            
            return JsonResponse({
                'success': True,
                'message': 'Video streaming stopped successfully'
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error stopping stream: {str(e)}'
            })
    else:
        return JsonResponse({
            'success': False,
            'error': 'No active stream found for this session'
        })

@require_http_methods(["GET"])
def stream_status(request, session_id):
    """Get current streaming status."""
    global active_streams
    
    is_streaming = session_id in active_streams
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    status_data = {
        'is_streaming': is_streaming,
        'session_status': session.status,
        'streaming_available': StreamingCarDetector is not None,
        'session_name': session.name
    }
    
    if is_streaming:
        detector = active_streams[session_id]
        status_data.update({
            'frames_processed': detector.frames_processed,
            'current_fps': detector.current_fps,
            'cars_detected': len(detector.global_car_db)
        })
    
    return JsonResponse(status_data)

@require_http_methods(["GET"])
def stream_frame(request, session_id):
    """Get current frame as image (fallback for non-WebSocket)."""
    frame_base64 = cache.get(f'current_frame_{session_id}')
    
    if frame_base64:
        try:
            frame_data = base64.b64decode(frame_base64)
            return HttpResponse(frame_data, content_type='image/jpeg')
        except Exception as e:
            print(f"Frame decode error: {e}")
    
    # Return placeholder image
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No stream available", (200, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(placeholder, "Click 'Start Stream' to begin", (180, 280),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', placeholder)
    return HttpResponse(buffer.tobytes(), content_type='image/jpeg')

# ==================================================
# SESSION MANAGEMENT AND UTILITY VIEWS
# ==================================================

def session_progress(request, session_id):
    """Get session progress via AJAX."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    progress_percentage = 0
    if session.total_frames > 0:
        progress_percentage = (session.processed_frames / session.total_frames) * 100
    
    return JsonResponse({
        'status': session.status,
        'processed_frames': session.processed_frames,
        'total_frames': session.total_frames,
        'progress_percentage': round(progress_percentage, 2),
        'cars_detected': session.cars_detected
    })

def export_session_data(request, session_id):
    """Export session data as JSON."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    cars = session.cars.all()
    
    export_data = {
        'session': {
            'id': session.id,
            'name': session.name,
            'status': session.status,
            'created_at': session.created_at.isoformat(),
            'cars_detected': session.cars_detected,
            'total_frames': session.total_frames,
            'processed_frames': session.processed_frames,
            'motion_threshold': session.motion_threshold,
            'motion_sensitivity': session.motion_sensitivity,
            'min_motion_area': session.min_motion_area,
            'consecutive_motion_frames': session.consecutive_motion_frames,
            'k_neighbors': session.k_neighbors,
            'location_precision': session.location_precision
        },
        'cars': []
    }
    
    for car in cars:
        car_data = {
            'car_id': car.car_id,
            'total_detections': car.total_detections,
            'max_speed': float(car.max_speed),
            'avg_speed': float(car.avg_speed),
            'total_distance': float(car.total_distance),
            'first_detected': car.first_detected.isoformat(),
            'last_seen': car.last_seen.isoformat(),
            'is_active': car.is_active,
            'tracking_points': []
        }
        
        for point in car.tracking_points.all():
            car_data['tracking_points'].append({
                'timestamp': point.timestamp.isoformat(),
                'latitude': float(point.latitude),
                'longitude': float(point.longitude),
                'pixel_x': float(point.pixel_x),
                'pixel_y': float(point.pixel_y),
                'speed': float(point.speed),
                'is_moving': point.is_moving,
                'motion_confidence': float(point.motion_confidence),
                'bbox_x1': float(point.bbox_x1),
                'bbox_y1': float(point.bbox_y1),
                'bbox_x2': float(point.bbox_x2),
                'bbox_y2': float(point.bbox_y2)
            })
        
        export_data['cars'].append(car_data)
    
    response = JsonResponse(export_data, json_dumps_params={'indent': 2})
    response['Content-Disposition'] = f'attachment; filename="session_{session.id}_data.json"'
    return response

def duplicate_session(request, session_id):
    """Duplicate an existing session with same settings."""
    if request.method == 'POST':
        original_session = get_object_or_404(CarTrackingSession, id=session_id)
        
        # Create new session with same settings
        new_session = CarTrackingSession.objects.create(
            name=f"{original_session.name} (Copy)",
            video_file=original_session.video_file,
            polygon_file=original_session.polygon_file,
            geo_coords_file=original_session.geo_coords_file,
            reference_image=original_session.reference_image,
            motion_threshold=original_session.motion_threshold,
            motion_sensitivity=original_session.motion_sensitivity,
            min_motion_area=original_session.min_motion_area,
            consecutive_motion_frames=original_session.consecutive_motion_frames,
            k_neighbors=original_session.k_neighbors,
            location_precision=original_session.location_precision,
            status='pending'
        )
        
        messages.success(request, f'Session duplicated successfully as "{new_session.name}"')
        return JsonResponse({
            'success': True,
            'new_session_id': new_session.id,
            'message': f'Session duplicated as "{new_session.name}"'
        })
    
    return JsonResponse({'error': 'Invalid request method'})

def reset_session(request, session_id):
    """Reset a session to pending status and clear all data."""
    if request.method == 'POST':
        session = get_object_or_404(CarTrackingSession, id=session_id)
        
        # Stop streaming if active
        global active_streams
        if session_id in active_streams:
            active_streams[session_id].stop_streaming()
            del active_streams[session_id]
        
        # Delete all related data
        session.cars.all().delete()  # This will cascade delete tracking points
        session.detection_logs.all().delete()
        
        # Reset session status and counters
        session.status = 'pending'
        session.processed_frames = 0
        session.total_frames = 0
        session.cars_detected = 0
        session.save()
        
        messages.success(request, f'Session "{session.name}" has been reset successfully.')
        return JsonResponse({
            'success': True,
            'message': f'Session "{session.name}" has been reset successfully.'
        })
    
    return JsonResponse({'error': 'Invalid request method'})

def delete_session(request, session_id):
    """Delete a tracking session."""
    if request.method == 'POST':
        session = get_object_or_404(CarTrackingSession, id=session_id)
        session_name = session.name
        
        # Stop streaming if active
        global active_streams
        if session_id in active_streams:
            active_streams[session_id].stop_streaming()
            del active_streams[session_id]
        
        # Delete associated files
        try:
            if session.video_file and default_storage.exists(session.video_file.name):
                default_storage.delete(session.video_file.name)
            if session.polygon_file and default_storage.exists(session.polygon_file.name):
                default_storage.delete(session.polygon_file.name)
            if session.geo_coords_file and default_storage.exists(session.geo_coords_file.name):
                default_storage.delete(session.geo_coords_file.name)
            if session.reference_image and default_storage.exists(session.reference_image.name):
                default_storage.delete(session.reference_image.name)
        except Exception as e:
            print(f"Could not delete some files: {str(e)}")
        
        session.delete()
        messages.success(request, f'Session "{session_name}" deleted successfully.')
        return JsonResponse({
            'success': True,
            'message': f'Session "{session_name}" deleted successfully.'
        })
        
    return JsonResponse({'error': 'Invalid request method'})

# ==================================================
# DEBUG AND TESTING VIEWS
# ==================================================

@require_http_methods(["GET"])
def debug_detection_accuracy(request, session_id):
    """Debug view to compare detection settings."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    debug_info = {
        'session_id': session_id,
        'session_name': session.name,
        'motion_settings': {
            'threshold': session.motion_threshold,
            'sensitivity': session.motion_sensitivity,
            'min_area': session.min_motion_area,
            'consecutive_frames': session.consecutive_motion_frames,
        },
        'coordinate_settings': {
            'k_neighbors': session.k_neighbors,
            'location_precision': session.location_precision,
        },
        'yolo_settings': {
            'confidence': 0.3,
            'image_size': 1088,
            'classes': [2],  # Cars only
        },
        'recommendations': []
    }
    
    # Add recommendations based on detection issues
    if session.motion_threshold > 30:
        debug_info['recommendations'].append({
            'issue': 'Motion threshold too high',
            'suggestion': 'Lower motion_threshold to 30 or below for better sensitivity',
            'current': session.motion_threshold,
            'recommended': 30
        })
    
    if session.motion_sensitivity > 0.15:
        debug_info['recommendations'].append({
            'issue': 'Motion sensitivity too high',
            'suggestion': 'Lower motion_sensitivity to 0.1 for better detection',
            'current': session.motion_sensitivity,
            'recommended': 0.1
        })
    
    if session.consecutive_motion_frames > 3:
        debug_info['recommendations'].append({
            'issue': 'Too many consecutive frames required',
            'suggestion': 'Lower consecutive_motion_frames to 3 for faster detection',
            'current': session.consecutive_motion_frames,
            'recommended': 3
        })
    
    return JsonResponse(debug_info, json_dumps_params={'indent': 2})

def debug_streaming(request, session_id):
    """Debug streaming setup."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    debug_info = {
        'session_id': session_id,
        'session_name': session.name,
        'session_status': session.status,
        'video_file_exists': bool(session.video_file and session.video_file.name),
        'polygon_file_exists': bool(session.polygon_file and session.polygon_file.name),
        'streaming_detector_available': StreamingCarDetector is not None,
    }
    
    # Check video file details
    if session.video_file:
        try:
            import os
            debug_info['video_file_path'] = session.video_file.path
            debug_info['video_file_url'] = session.video_file.url
            
            if os.path.exists(session.video_file.path):
                cap = cv2.VideoCapture(session.video_file.path)
                debug_info['video_can_open'] = cap.isOpened()
                if cap.isOpened():
                    debug_info['video_frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    debug_info['video_fps'] = cap.get(cv2.CAP_PROP_FPS)
                    debug_info['video_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    debug_info['video_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            else:
                debug_info['video_error'] = 'Video file path does not exist'
        except Exception as e:
            debug_info['video_error'] = str(e)
    else:
        debug_info['video_error'] = 'No video file uploaded'
    
    # Test YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        debug_info['yolo_model_loaded'] = True
        debug_info['yolo_model_path'] = "yolo11n.pt"
    except Exception as e:
        debug_info['yolo_error'] = str(e)
        debug_info['yolo_model_loaded'] = False
    
    # Test coordinate converter
    try:
        from .coordinate_converter import CoordinateConverter
        debug_info['coordinate_converter_available'] = True
    except Exception as e:
        debug_info['coordinate_converter_error'] = str(e)
        debug_info['coordinate_converter_available'] = False
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        r.ping()
        debug_info['redis_available'] = True
    except Exception as e:
        debug_info['redis_error'] = str(e)
        debug_info['redis_available'] = False
    
    # Check WebSocket channels
    try:
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        debug_info['channels_available'] = channel_layer is not None
    except Exception as e:
        debug_info['channels_error'] = str(e)
        debug_info['channels_available'] = False
    
    return JsonResponse(debug_info, json_dumps_params={'indent': 2})

def test_streaming(request, session_id):
    """Test if streaming can start for a session."""
    session = get_object_or_404(CarTrackingSession, id=session_id)
    
    if StreamingCarDetector is None:
        return JsonResponse({
            'success': False,
            'error': 'StreamingCarDetector not available'
        })
    
    try:
        detector = StreamingCarDetector(session_id)
        success, message = detector.start_streaming()
        
        if success:
            # Stop it immediately after testing
            detector.stop_streaming()
        
        return JsonResponse({
            'success': success,
            'message': message,
            'session_id': session_id
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'session_id': session_id
        })

# ==================================================
# CLEANUP FUNCTION
# ==================================================

def cleanup_inactive_streams():
    """Clean up inactive streaming sessions."""
    global active_streams
    
    inactive_sessions = []
    for session_id, detector in active_streams.items():
        if not detector.is_streaming:
            inactive_sessions.append(session_id)
    
    for session_id in inactive_sessions:
        if session_id in active_streams:
            active_streams[session_id].stop_streaming()
            del active_streams[session_id]
            print(f"🧹 Cleaned up inactive stream: {session_id}")

# Run cleanup periodically
import atexit
atexit.register(cleanup_inactive_streams)