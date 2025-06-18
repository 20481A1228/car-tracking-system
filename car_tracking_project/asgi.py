# car_tracking_project/asgi.py - CORRECT FIXED VERSION

import os
import django
from django.core.asgi import get_asgi_application

# CRITICAL: Set Django settings FIRST, before any imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'car_tracking_project.settings')

# Setup Django BEFORE importing anything that uses models
django.setup()

# NOW we can safely import channels and routing
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import tracking.routing

# Get the Django ASGI application (for HTTP requests)
django_asgi_app = get_asgi_application()

# Create the ASGI application with WebSocket support
application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            tracking.routing.websocket_urlpatterns
        )
    ),
})

print("✅ ASGI application configured with WebSocket support")