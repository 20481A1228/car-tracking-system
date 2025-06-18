from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count, Avg, Max
from tracking.models import CarTrackingSession, Car, TrackingPoint, DetectionLog
from .serializers import (CarTrackingSessionSerializer, CarSerializer, 
                         TrackingPointSerializer, DetectionLogSerializer)

class CarTrackingSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing car tracking sessions."""
    queryset = CarTrackingSession.objects.all()
    serializer_class = CarTrackingSessionSerializer
    
    @action(detail=True, methods=['get'])
    def stats(self, request, pk=None):
        """Get session statistics."""
        session = self.get_object()
        stats = {
            'total_cars': session.cars.count(),
            'active_cars': session.cars.filter(is_active=True).count(),
            'total_tracking_points': TrackingPoint.objects.filter(car__session=session).count(),
            'avg_speed': session.cars.aggregate(avg_speed=Avg('avg_speed'))['avg_speed'] or 0,
            'max_speed': session.cars.aggregate(max_speed=Max('max_speed'))['max_speed'] or 0,
            'status': session.status,
            'progress': {
                'processed_frames': session.processed_frames,
                'total_frames': session.total_frames,
                'percentage': (session.processed_frames / session.total_frames * 100) if session.total_frames else 0
            }
        }
        return Response(stats)

class CarViewSet(viewsets.ModelViewSet):
    """ViewSet for managing cars."""
    queryset = Car.objects.all()
    serializer_class = CarSerializer
    
    def get_queryset(self):
        """Filter cars by session if provided."""
        queryset = Car.objects.all()
        session_id = self.request.query_params.get('session', None)
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        return queryset
    
    @action(detail=True, methods=['get'])
    def path(self, request, pk=None):
        """Get complete tracking path for a car."""
        car = self.get_object()
        points = car.tracking_points.all().order_by('timestamp')
        serializer = TrackingPointSerializer(points, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def live_positions(self, request):
        """Get current positions of all active cars."""
        session_id = request.query_params.get('session')
        if not session_id:
            return Response({'error': 'Session ID required'}, status=400)
        
        cars = Car.objects.filter(session_id=session_id, is_active=True)
        live_data = []
        
        for car in cars:
            latest_point = car.tracking_points.last()
            if latest_point:
                live_data.append({
                    'car_id': car.car_id,
                    'latitude': latest_point.latitude,
                    'longitude': latest_point.longitude,
                    'speed': latest_point.speed,
                    'timestamp': latest_point.timestamp,
                    'motion_confidence': latest_point.motion_confidence
                })
        
        return Response(live_data)

class TrackingPointViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for tracking points (read-only)."""
    queryset = TrackingPoint.objects.all()
    serializer_class = TrackingPointSerializer
    
    def get_queryset(self):
        """Filter tracking points by car or session."""
        queryset = TrackingPoint.objects.all()
        car_id = self.request.query_params.get('car_id', None)
        session_id = self.request.query_params.get('session', None)
        
        if car_id:
            queryset = queryset.filter(car_id=car_id)
        if session_id:
            queryset = queryset.filter(car__session_id=session_id)
            
        return queryset.order_by('timestamp')

class DetectionLogViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for detection logs (read-only)."""
    queryset = DetectionLog.objects.all()
    serializer_class = DetectionLogSerializer
    
    def get_queryset(self):
        """Filter logs by session."""
        queryset = DetectionLog.objects.all()
        session_id = self.request.query_params.get('session', None)
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        return queryset.order_by('-timestamp')

print("✅ API Views file created!")
print("📁 Save this as: api/views.py")
print("🎯 This will fix the ViewSet import error!")