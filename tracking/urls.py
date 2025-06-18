# tracking/urls.py - COMPLETE URL CONFIGURATION WITH STREAMING ENDPOINTS

from django.urls import path
from . import views

urlpatterns = [
    # ==================================================
    # MAIN DASHBOARD AND SESSION URLS
    # ==================================================
    
    # Main dashboard and core pages
    path('', views.dashboard, name='dashboard'),
    path('create/', views.create_session, name='create_session'),
    path('analytics/', views.analytics, name='analytics'),
    
    # Session-related URLs
    path('session/<int:session_id>/', views.session_detail, name='session_detail'),
    path('session/<int:session_id>/start/', views.start_processing, name='start_processing'),
    path('session/<int:session_id>/visualization/', views.visualization, name='visualization'),
    path('session/<int:session_id>/live/', views.live_data, name='live_data'),
    
    # ==================================================
    # LIVE STREAMING URLS (NEW)
    # ==================================================
    
    # Live streaming URLs
    path('session/<int:session_id>/stream/', views.live_stream_view, name='live_stream'),
    path('session/<int:session_id>/start-streaming/', views.start_video_streaming, name='start_video_streaming'),
    path('session/<int:session_id>/stop-streaming/', views.stop_video_streaming, name='stop_video_streaming'),
    path('session/<int:session_id>/stream-frame/', views.stream_frame, name='stream_frame'),
    path('session/<int:session_id>/stream-status/', views.stream_status, name='stream_status'),
    
    # ==================================================
    # SESSION MANAGEMENT AND UTILITY URLS
    # ==================================================
    
    # Session management and utility URLs
    path('session/<int:session_id>/progress/', views.session_progress, name='session_progress'),
    path('session/<int:session_id>/export/', views.export_session_data, name='export_session_data'),
    path('session/<int:session_id>/duplicate/', views.duplicate_session, name='duplicate_session'),
    path('session/<int:session_id>/reset/', views.reset_session, name='reset_session'),
    path('session/<int:session_id>/delete/', views.delete_session, name='delete_session'),
    
    # ==================================================
    # CAR-RELATED URLS
    # ==================================================
    
    # Car-related URLs
    path('car/<int:car_id>/', views.car_detail, name='car_detail'),
    path('debug/<int:session_id>/', views.debug_streaming, name='debug_streaming'),
    path('debug/<int:session_id>/', views.debug_streaming, name='debug_streaming'),
    path('test-streaming/<int:session_id>/', views.test_streaming, name='test_streaming'),
]