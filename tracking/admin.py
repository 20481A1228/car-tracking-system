from django.contrib import admin
from .models import CarTrackingSession, Car, TrackingPoint, DetectionLog, ReferencePoint

@admin.register(CarTrackingSession)
class CarTrackingSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'status', 'cars_detected', 'processed_frames', 'total_frames', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['name']
    readonly_fields = ['status', 'processed_frames', 'total_frames', 'cars_detected', 'created_at', 'updated_at']

@admin.register(Car)
class CarAdmin(admin.ModelAdmin):
    list_display = ['car_id', 'session', 'total_detections', 'max_speed', 'is_active', 'first_detected', 'last_seen']
    list_filter = ['session', 'is_active', 'first_detected']
    search_fields = ['car_id', 'session__name']
    readonly_fields = ['first_detected', 'last_seen']

@admin.register(TrackingPoint)
class TrackingPointAdmin(admin.ModelAdmin):
    list_display = ['car', 'timestamp', 'latitude', 'longitude', 'speed', 'is_moving', 'motion_confidence']
    list_filter = ['is_moving', 'timestamp', 'car__session']
    search_fields = ['car__car_id']
    readonly_fields = ['timestamp']

@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ['session', 'level', 'message', 'timestamp']
    list_filter = ['level', 'timestamp', 'session']
    search_fields = ['message', 'session__name']
    readonly_fields = ['timestamp']

@admin.register(ReferencePoint)
class ReferencePointAdmin(admin.ModelAdmin):
    list_display = ['session', 'point_order', 'latitude', 'longitude', 'pixel_x', 'pixel_y']
    list_filter = ['session']
    ordering = ['session', 'point_order']