# car_tracking_project/settings.py - COMPLETE PRODUCTION-READY VERSION

"""
Django settings for car_tracking_project.
Complete production-ready configuration for Car Tracking System with Live Video Streaming.
"""

import os
import sys
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# ==================================================
# ENVIRONMENT VARIABLES & CONFIGURATION
# ==================================================

# Try to import python-decouple for environment variables
try:
    from decouple import config, Csv
    print("✅ Using python-decouple for environment variables")
except ImportError:
    print("⚠️ python-decouple not installed, using os.environ")
    class MockConfig:
        def __call__(self, key, default=None, cast=str):
            value = os.environ.get(key, default)
            if cast == bool:
                return value.lower() in ('true', '1', 'yes', 'on') if isinstance(value, str) else bool(value)
            elif cast == Csv:
                return value.split(',') if value else []
            return cast(value) if value is not None else default
    config = MockConfig()
    Csv = lambda: list

# ==================================================
# SECURITY SETTINGS
# ==================================================

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config(
    'SECRET_KEY', 
    default='django-insecure-car-tracking-change-this-in-production-abc123xyz'
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

# Allowed hosts
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1', cast=Csv())

# Add any additional hosts for deployment
if not DEBUG:
    # Add common deployment hosts
    additional_hosts = [
        '.herokuapp.com',      # Heroku
        '.railway.app',        # Railway
        '.vercel.app',         # Vercel
        '.onrender.com',       # Render
        '.digitalocean.com',   # DigitalOcean
    ]
    ALLOWED_HOSTS.extend(additional_hosts)

# ==================================================
# APPLICATION DEFINITION
# ==================================================

INSTALLED_APPS = [
    # Django core apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'channels',  # For WebSocket support
    'corsheaders',  # For API CORS
    
    # Local apps
    'tracking',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # For static files in production
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'car_tracking_project.urls'

# ==================================================
# ASGI CONFIGURATION (FOR WEBSOCKET SUPPORT)
# ==================================================

ASGI_APPLICATION = 'car_tracking_project.asgi.application'

# ==================================================
# TEMPLATES CONFIGURATION
# ==================================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'car_tracking_project.wsgi.application'

# ==================================================
# DATABASE CONFIGURATION
# ==================================================

# Check for DATABASE_URL (production)
DATABASE_URL = config('DATABASE_URL', default=None)

if DATABASE_URL:
    # Production database (PostgreSQL)
    try:
        import dj_database_url
        DATABASES = {
            'default': dj_database_url.parse(DATABASE_URL)
        }
        print("✅ Using PostgreSQL database from DATABASE_URL")
    except ImportError:
        print("⚠️ dj_database_url not installed, falling back to SQLite")
        DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': BASE_DIR / 'car_tracking.db',
            }
        }
else:
    # Development database (SQLite)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'car_tracking.db',
        }
    }
    print("✅ Using SQLite database for development")

# ==================================================
# REDIS CONFIGURATION
# ==================================================

# Redis URL configuration
REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')

# Parse Redis URL for different components
if REDIS_URL.startswith('rediss://'):
    # SSL Redis (common in production)
    REDIS_SSL = True
    REDIS_CONNECTION = {
        'host': REDIS_URL.split('@')[1].split(':')[0],
        'port': int(REDIS_URL.split(':')[-1].split('/')[0]),
        'db': int(REDIS_URL.split('/')[-1]),
        'ssl_cert_reqs': None,
    }
elif REDIS_URL.startswith('redis://'):
    # Standard Redis
    REDIS_SSL = False
    REDIS_CONNECTION = REDIS_URL
else:
    # Fallback
    REDIS_CONNECTION = 'redis://localhost:6379/0'

print(f"✅ Redis configured: {REDIS_URL}")

# ==================================================
# CHANNELS CONFIGURATION (ENHANCED FOR STABILITY)
# ==================================================

try:
    import redis
    import channels_redis
    
    # Test Redis connection
    if isinstance(REDIS_CONNECTION, dict):
        r = redis.Redis(**REDIS_CONNECTION)
    else:
        r = redis.from_url(REDIS_CONNECTION)
    r.ping()
    
    # Redis is available
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                "hosts": [REDIS_CONNECTION],
                "capacity": 1500,  # Max messages in channel
                "expiry": 300,     # Message expiry (5 minutes)
                "group_expiry": 86400,  # Group expiry (24 hours)
                "channel_capacity": {
                    "http.request": 200,
                    "http.response": 200,
                    "websocket.connect": 100,
                    "websocket.accept": 100,
                    "websocket.receive": 100,
                    "websocket.send": 100,
                    "websocket.disconnect": 100,
                },
            },
        },
    }
    print("✅ Redis channel layer configured successfully")
    
except (ImportError, redis.ConnectionError, redis.ResponseError) as e:
    # Fallback to in-memory channel layer
    print(f"⚠️ Redis not available ({e}), using in-memory channel layer")
    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels.layers.InMemoryChannelLayer',
            'CONFIG': {
                "capacity": 1500,
                "expiry": 300,
            }
        }
    }

# ==================================================
# WEBSOCKET STABILITY SETTINGS
# ==================================================

# WebSocket connection settings
WEBSOCKET_ACCEPT_ALL = True
WEBSOCKET_TIMEOUT = config('WEBSOCKET_TIMEOUT', default=300, cast=int)  # 5 minutes
WEBSOCKET_CLOSE_TIMEOUT = 10  # 10 seconds
WEBSOCKET_HEARTBEAT_INTERVAL = 30  # 30 seconds
WEBSOCKET_MAX_RECONNECT_ATTEMPTS = 5

# ==================================================
# CACHING CONFIGURATION
# ==================================================

try:
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.redis.RedisCache',
            'LOCATION': REDIS_CONNECTION,
            'OPTIONS': {
                'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            } if 'django_redis' in sys.modules else {},
            'KEY_PREFIX': 'car_tracking',
            'TIMEOUT': 300,  # 5 minutes default timeout
        }
    }
    print("✅ Redis cache configured")
except Exception as e:
    # Fallback to local memory cache
    print(f"⚠️ Redis cache not available ({e}), using local memory cache")
    CACHES = {
        'default': {
            'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
            'LOCATION': 'car-tracking-cache',
        }
    }

# ==================================================
# CELERY CONFIGURATION (FOR BACKGROUND TASKS)
# ==================================================

CELERY_BROKER_URL = config('CELERY_BROKER_URL', default=REDIS_URL)
CELERY_RESULT_BACKEND = config('CELERY_RESULT_BACKEND', default=REDIS_URL)
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# Celery task routes
CELERY_TASK_ROUTES = {
    'tracking.tasks.process_video_async': {'queue': 'video_processing'},
    'tracking.tasks.cleanup_old_sessions': {'queue': 'maintenance'},
}

# Celery beat schedule
CELERY_BEAT_SCHEDULE = {
    'cleanup-old-sessions': {
        'task': 'tracking.tasks.cleanup_old_sessions',
        'schedule': 3600.0,  # Every hour
    },
}

print(f"✅ Celery configured with broker: {CELERY_BROKER_URL}")

# ==================================================
# STATIC FILES CONFIGURATION
# ==================================================

STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Static files finders
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

# WhiteNoise configuration for production
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# ==================================================
# MEDIA FILES CONFIGURATION
# ==================================================

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = config('FILE_UPLOAD_MAX_MEMORY_SIZE', default=500 * 1024 * 1024, cast=int)  # 500MB
DATA_UPLOAD_MAX_MEMORY_SIZE = config('DATA_UPLOAD_MAX_MEMORY_SIZE', default=500 * 1024 * 1024, cast=int)  # 500MB
FILE_UPLOAD_PERMISSIONS = 0o644

# Create media directories automatically
MEDIA_SUBDIRS = ['videos', 'polygons', 'geo_coords', 'reference_images']
for subdir in MEDIA_SUBDIRS:
    subdir_path = MEDIA_ROOT / subdir
    subdir_path.mkdir(parents=True, exist_ok=True)

# ==================================================
# AWS S3 CONFIGURATION (OPTIONAL FOR PRODUCTION)
# ==================================================

USE_S3 = config('USE_S3', default=False, cast=bool)

if USE_S3:
    try:
        import boto3
        from storages.backends.s3boto3 import S3Boto3Storage
        
        # AWS settings
        AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
        AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
        AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME', default='us-east-1')
        AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'
        AWS_DEFAULT_ACL = None
        AWS_S3_OBJECT_PARAMETERS = {
            'CacheControl': 'max-age=86400',
        }
        
        # Static and media files storage
        STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
        DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
        
        STATIC_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/static/'
        MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/media/'
        
        print("✅ AWS S3 storage configured")
        
    except ImportError:
        print("⚠️ boto3 not installed, using local file storage")
        USE_S3 = False

# ==================================================
# PASSWORD VALIDATION
# ==================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# ==================================================
# INTERNATIONALIZATION
# ==================================================

LANGUAGE_CODE = 'en-us'
TIME_ZONE = config('TIME_ZONE', default='UTC')
USE_I18N = True
USE_TZ = True

# ==================================================
# REST FRAMEWORK CONFIGURATION
# ==================================================

REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': config('API_PAGE_SIZE', default=100, cast=int),
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # Change in production with auth
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': config('API_THROTTLE_ANON', default='100/hour'),
        'user': config('API_THROTTLE_USER', default='1000/hour')
    }
}

# ==================================================
# CORS CONFIGURATION
# ==================================================

CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default='http://localhost:3000,http://127.0.0.1:3000', cast=Csv())

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_ALL_ORIGINS = DEBUG  # Allow all origins in development

# ==================================================
# CAR TRACKING SPECIFIC SETTINGS
# ==================================================

CAR_TRACKING = {
    # Video processing settings
    'FRAME_WIDTH': config('FRAME_WIDTH', default=640, cast=int),
    'FRAME_HEIGHT': config('FRAME_HEIGHT', default=360, cast=int),
    'FRAME_SKIP': config('FRAME_SKIP', default=3, cast=int),  # Process every Nth frame
    
    # Motion detection settings (can be overridden per session)
    'DEFAULT_MOTION_THRESHOLD': config('DEFAULT_MOTION_THRESHOLD', default=30, cast=int),
    'DEFAULT_MOTION_SENSITIVITY': config('DEFAULT_MOTION_SENSITIVITY', default=0.1, cast=float),
    'DEFAULT_MIN_MOTION_AREA': config('DEFAULT_MIN_MOTION_AREA', default=100, cast=int),
    'DEFAULT_CONSECUTIVE_MOTION_FRAMES': config('DEFAULT_CONSECUTIVE_MOTION_FRAMES', default=3, cast=int),
    
    # Coordinate conversion settings
    'DEFAULT_K_NEIGHBORS': config('DEFAULT_K_NEIGHBORS', default=4, cast=int),
    'DEFAULT_LOCATION_PRECISION': config('DEFAULT_LOCATION_PRECISION', default=6, cast=int),
    
    # YOLO model settings
    'YOLO_MODEL': config('YOLO_MODEL', default='yolo11n.pt'),
    'YOLO_CONFIDENCE': config('YOLO_CONFIDENCE', default=0.1, cast=float),
    'YOLO_IMAGE_SIZE': config('YOLO_IMAGE_SIZE', default=640, cast=int),
    'DETECTION_CLASSES': [2],  # Only detect cars (class 2 in COCO dataset)
    
    # File validation settings
    'ALLOWED_VIDEO_FORMATS': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
    'ALLOWED_IMAGE_FORMATS': ['.jpg', '.jpeg', '.png', '.bmp'],
    'MAX_VIDEO_SIZE': config('MAX_VIDEO_SIZE', default=500 * 1024 * 1024, cast=int),  # 500MB
    'MAX_IMAGE_SIZE': config('MAX_IMAGE_SIZE', default=50 * 1024 * 1024, cast=int),   # 50MB
    
    # Streaming settings
    'STREAMING_FPS': config('STREAMING_FPS', default=10, cast=int),
    'STREAMING_QUALITY': config('STREAMING_QUALITY', default=80, cast=int),  # JPEG quality (1-100)
    'MAX_STREAMING_SESSIONS': config('MAX_STREAMING_SESSIONS', default=5, cast=int),
    'STREAMING_TIMEOUT': config('STREAMING_TIMEOUT', default=300, cast=int),  # 5 minutes
}

# ==================================================
# LOGGING CONFIGURATION
# ==================================================

# Create logs directory
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
        'streaming': {
            'format': '[STREAM] {levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'car_tracking.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'streaming_file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'streaming.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 3,
            'formatter': 'streaming',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'tracking': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'streaming': {
            'handlers': ['console', 'streaming_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'channels': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# ==================================================
# SESSION CONFIGURATION
# ==================================================

SESSION_ENGINE = 'django.contrib.sessions.backends.cached_db'
SESSION_COOKIE_AGE = config('SESSION_COOKIE_AGE', default=86400, cast=int)  # 24 hours
SESSION_SAVE_EVERY_REQUEST = True
SESSION_EXPIRE_AT_BROWSER_CLOSE = False

# ==================================================
# SECURITY SETTINGS
# ==================================================

# CSRF settings
CSRF_COOKIE_SECURE = config('CSRF_COOKIE_SECURE', default=not DEBUG, cast=bool)
CSRF_COOKIE_HTTPONLY = True
CSRF_TRUSTED_ORIGINS = config('CSRF_TRUSTED_ORIGINS', default='http://localhost:8000,http://127.0.0.1:8000', cast=Csv())

# Security headers
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Production security settings
if not DEBUG:
    SECURE_SSL_REDIRECT = config('SECURE_SSL_REDIRECT', default=False, cast=bool)
    SECURE_HSTS_SECONDS = config('SECURE_HSTS_SECONDS', default=31536000, cast=int)  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = config('SECURE_HSTS_INCLUDE_SUBDOMAINS', default=True, cast=bool)
    SECURE_HSTS_PRELOAD = config('SECURE_HSTS_PRELOAD', default=True, cast=bool)
    SESSION_COOKIE_SECURE = config('SESSION_COOKIE_SECURE', default=True, cast=bool)
    CSRF_COOKIE_SECURE = True

# ==================================================
# EMAIL CONFIGURATION
# ==================================================

if DEBUG:
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
else:
    EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
    EMAIL_HOST = config('EMAIL_HOST', default='smtp.gmail.com')
    EMAIL_PORT = config('EMAIL_PORT', default=587, cast=int)
    EMAIL_USE_TLS = config('EMAIL_USE_TLS', default=True, cast=bool)
    EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='')
    EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')

# ==================================================
# MESSAGES FRAMEWORK
# ==================================================

from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.DEBUG: 'debug',
    messages.INFO: 'info',
    messages.SUCCESS: 'success',
    messages.WARNING: 'warning',
    messages.ERROR: 'danger',
}

# ==================================================
# DEFAULT PRIMARY KEY FIELD TYPE
# ==================================================

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ==================================================
# DEVELOPMENT TOOLS (DEBUG MODE ONLY)
# ==================================================

if DEBUG:
    INTERNAL_IPS = ['127.0.0.1', 'localhost']
    
    # Optional: Django Debug Toolbar (install separately)
    try:
        import debug_toolbar
        INSTALLED_APPS.append('debug_toolbar')
        MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
        DEBUG_TOOLBAR_CONFIG = {
            'SHOW_TOOLBAR_CALLBACK': lambda request: DEBUG,
        }
        print("✅ Debug toolbar enabled")
    except ImportError:
        pass

# ==================================================
# STREAMING VALIDATION & SETUP
# ==================================================

def validate_streaming_requirements():
    """Validate that all streaming requirements are available."""
    missing_requirements = []
    available_features = []
    
    # Check OpenCV
    try:
        import cv2
        available_features.append(f'OpenCV {cv2.__version__}')
    except ImportError:
        missing_requirements.append('opencv-python')
    
    # Check NumPy
    try:
        import numpy
        available_features.append(f'NumPy {numpy.__version__}')
    except ImportError:
        missing_requirements.append('numpy')
    
    # Check YOLO
    try:
        from ultralytics import YOLO
        available_features.append('YOLO (Ultralytics)')
    except ImportError:
        missing_requirements.append('ultralytics')
    
    # Check Channels
    try:
        import channels
        available_features.append(f'Channels {channels.__version__}')
    except ImportError:
        missing_requirements.append('channels')
    
    # Check Redis
    try:
        import redis
        available_features.append(f'Redis client {redis.__version__}')
    except ImportError:
        missing_requirements.append('redis')
    
    if missing_requirements:
        print(f"⚠️ Missing streaming requirements: {', '.join(missing_requirements)}")
        print(f"   Install with: pip install {' '.join(missing_requirements)}")
        return False
    else:
        print(f"✅ All streaming requirements available: {', '.join(available_features)}")
        return True

# Check streaming requirements on startup
STREAMING_AVAILABLE = validate_streaming_requirements()

# ==================================================
# HEROKU-SPECIFIC CONFIGURATION
# ==================================================

# Heroku deployment detection
if 'DYNO' in os.environ:
    print("🚀 Heroku deployment detected")
    
    # Use Heroku's PORT
    PORT = int(os.environ.get('PORT', 8000))
    
    # Heroku Redis
    if 'REDIS_URL' in os.environ:
        REDIS_URL = os.environ['REDIS_URL']
        CELERY_BROKER_URL = REDIS_URL
        CELERY_RESULT_BACKEND = REDIS_URL
    
    # Security settings for Heroku
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = True

# ==================================================
# RAILWAY-SPECIFIC CONFIGURATION
# ==================================================

# Railway deployment detection
if 'RAILWAY_ENVIRONMENT' in os.environ:
    print("🚀 Railway deployment detected")
    
    # Railway provides PORT
    PORT = int(os.environ.get('PORT', 8000))
    
    # Railway Redis
    if 'REDISURL' in os.environ:
        REDIS_URL = os.environ['REDISURL']
        CELERY_BROKER_URL = REDIS_URL
        CELERY_RESULT_BACKEND = REDIS_URL

# ==================================================
# RENDER-SPECIFIC CONFIGURATION
# ==================================================

# Render deployment detection
if 'RENDER' in os.environ:
    print("🚀 Render deployment detected")
    
    # Render Redis
    if 'REDIS_URL' in os.environ:
        REDIS_URL = os.environ['REDIS_URL']
        CELERY_BROKER_URL = REDIS_URL
        CELERY_RESULT_BACKEND = REDIS_URL

# ==================================================
# HEALTH CHECK ENDPOINT
# ==================================================

HEALTH_CHECK = {
    'ENABLED': True,
    'URL': '/health/',
    'CHECK_DATABASE': True,
    'CHECK_REDIS': True,
    'CHECK_STREAMING': True,
}

# ==================================================
# PRINT CONFIGURATION STATUS
# ==================================================

if DEBUG:
    print("\n🚗 Car Tracking System Configuration Loaded")
    print("="*60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Media Root: {MEDIA_ROOT}")
    print(f"🌐 Media URL: {MEDIA_URL}")
    print(f"📁 Static Root: {STATIC_ROOT}")
    print(f"🌐 Static URL: {STATIC_URL}")
    print(f"🔧 Debug Mode: {DEBUG}")
    print(f"📊 Database: {DATABASES['default']['ENGINE'].split('.')[-1].upper()}")
    print(f"🔌 ASGI Application: {ASGI_APPLICATION}")
    print(f"📡 Channel Layer: {CHANNEL_LAYERS['default']['BACKEND'].split('.')[-1]}")
    print(f"💾 Cache Backend: {CACHES['default']['BACKEND'].split('.')[-1]}")
    print(f"🎥 Streaming Available: {STREAMING_AVAILABLE}")
    print(f"⏰ WebSocket Timeout: {WEBSOCKET_TIMEOUT}s")
    print(f"💓 Heartbeat Interval: {WEBSOCKET_HEARTBEAT_INTERVAL}s")
    print(f"🌍 Allowed Hosts: {ALLOWED_HOSTS}")
    print(f"🔐 Secret Key: {'*' * 20}...{SECRET_KEY[-5:]}")
    print("="*60)
    print("✅ Configuration loaded successfully!")
    print("")
    
    if not STREAMING_AVAILABLE:
        print("⚠️ Some streaming requirements are missing.")
        print("   Install missing packages to enable full functionality.")
    
    print("🎯 Next Steps:")
    print("   1. Run: python manage.py collectstatic")
    print("   2. Run: python manage.py migrate")  
    print("   3. Start Redis: redis-server")
    print("   4. Run: daphne -b 127.0.0.1 -p 8000 car_tracking_project.asgi:application")
    print("   5. Access: http://localhost:8000/")
    print("")