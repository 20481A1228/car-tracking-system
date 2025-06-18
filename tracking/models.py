from django.db import models
from django.contrib.auth.models import User
import json

class CarTrackingSession(models.Model):
    name = models.CharField(max_length=200)
    video_file = models.FileField(upload_to='videos/')
    polygon_file = models.FileField(upload_to='polygons/')
    geo_coords_file = models.FileField(upload_to='geo_coords/', blank=True, null=True)
    reference_image = models.ImageField(upload_to='reference_images/', blank=True, null=True)
    
    # Enhanced motion detection settings
    motion_threshold = models.IntegerField(default=30)
    motion_sensitivity = models.FloatField(default=0.1)
    min_motion_area = models.IntegerField(default=100)
    consecutive_motion_frames = models.IntegerField(default=3)
    
    # Coordinate conversion settings
    k_neighbors = models.IntegerField(default=4)
    location_precision = models.IntegerField(default=6)
    
    # Session status
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Processing stats
    total_frames = models.IntegerField(default=0)
    processed_frames = models.IntegerField(default=0)
    cars_detected = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.name} - {self.status}"

class ReferencePoint(models.Model):
    session = models.ForeignKey(CarTrackingSession, on_delete=models.CASCADE, related_name='reference_points')
    pixel_x = models.FloatField()
    pixel_y = models.FloatField()
    yolo_x = models.FloatField()
    yolo_y = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    point_order = models.IntegerField()
    
    class Meta:
        ordering = ['point_order']

class Car(models.Model):
    session = models.ForeignKey(CarTrackingSession, on_delete=models.CASCADE, related_name='cars')
    car_id = models.CharField(max_length=50)
    first_detected = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    total_detections = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    # Stats
    max_speed = models.FloatField(default=0.0)
    avg_speed = models.FloatField(default=0.0)
    total_distance = models.FloatField(default=0.0)
    avg_motion_confidence = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"{self.car_id} ({self.session.name})"
    
    class Meta:
        unique_together = ['session', 'car_id']

class TrackingPoint(models.Model):
    car = models.ForeignKey(Car, on_delete=models.CASCADE, related_name='tracking_points')
    timestamp = models.DateTimeField()
    
    # Pixel coordinates
    pixel_x = models.FloatField()
    pixel_y = models.FloatField()
    
    # Geographic coordinates
    latitude = models.FloatField()
    longitude = models.FloatField()
    
    # Motion data
    is_moving = models.BooleanField(default=True)
    speed = models.FloatField(default=0.0)  # km/h
    motion_confidence = models.FloatField(default=0.0)  # 0-1
    
    # Bounding box
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    
    class Meta:
        ordering = ['timestamp']

class DetectionLog(models.Model):
    session = models.ForeignKey(
        CarTrackingSession, 
        on_delete=models.CASCADE, 
        related_name='detection_logs'  # CRITICAL: This creates session.detection_logs
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    message = models.TextField()
    level = models.CharField(max_length=10, choices=[
        ('INFO', 'Info'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
    ], default='INFO')
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.level}: {self.message[:50]}"
