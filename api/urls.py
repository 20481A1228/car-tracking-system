from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router and register our viewsets
router = DefaultRouter()
router.register(r'sessions', views.CarTrackingSessionViewSet)
router.register(r'cars', views.CarViewSet)
router.register(r'tracking-points', views.TrackingPointViewSet)
router.register(r'logs', views.DetectionLogViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

print("✅ API URLs file created!")
print("📁 Save this as: api/urls.py")
print("🎯 This properly registers all ViewSets with the router!")