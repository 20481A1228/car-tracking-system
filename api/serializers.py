from rest_framework import serializers
from tracking.models import CarTrackingSession, Car, TrackingPoint, DetectionLog, ReferencePoint

class ReferencePointSerializer(serializers.ModelSerializer):
    """Serializer for reference points."""
    class Meta:
        model = ReferencePoint
        fields = '__all__'

class CarTrackingSessionSerializer(serializers.ModelSerializer):
    """Serializer for car tracking sessions."""
    reference_points = ReferencePointSerializer(many=True, read_only=True)
    
    class Meta:
        model = CarTrackingSession
        fields = '__all__'

class TrackingPointSerializer(serializers.ModelSerializer):
    """Serializer for tracking points."""
    class Meta:
        model = TrackingPoint
        fields = ['id', 'timestamp', 'latitude', 'longitude', 'is_moving', 
                 'speed', 'motion_confidence', 'pixel_x', 'pixel_y']

class CarSerializer(serializers.ModelSerializer):
    """Serializer for cars."""
    tracking_points = TrackingPointSerializer(many=True, read_only=True)
    latest_position = serializers.SerializerMethodField()
    
    class Meta:
        model = Car
        fields = ['id', 'car_id', 'first_detected', 'last_seen', 'total_detections',
                 'max_speed', 'avg_speed', 'total_distance', 'is_active', 
                 'tracking_points', 'latest_position']
    
    def get_latest_position(self, obj):
        """Get the latest tracking point for this car."""
        latest = obj.tracking_points.last()
        if latest:
            return {
                'latitude': latest.latitude,
                'longitude': latest.longitude,
                'timestamp': latest.timestamp,
                'speed': latest.speed
            }
        return None

class DetectionLogSerializer(serializers.ModelSerializer):
    """Serializer for detection logs."""
    class Meta:
        model = DetectionLog
        fields = '__all__'

print("✅ API Serializers file created!")
print("📁 Save this as: api/serializers.py")
print("🎯 This provides the serializer classes needed by the views!")