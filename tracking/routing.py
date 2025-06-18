from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/tracking/(?P<session_id>\w+)/$', consumers.TrackingConsumer.as_asgi()),
    re_path(r'ws/stream/(?P<session_id>\w+)/$', consumers.VideoStreamConsumer.as_asgi()),
    re_path(r'ws/live-updates/$', consumers.LiveUpdatesConsumer.as_asgi()),
]