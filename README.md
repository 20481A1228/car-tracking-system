# 🚗 Car Tracking System

A sophisticated real-time car tracking system with live video streaming, built with Django, WebSocket, and computer vision.

## ✨ Features

- **Real-time Video Streaming** with YOLO car detection
- **Live WebSocket Updates** for instant car tracking
- **Interactive Mapping** with Leaflet.js
- **Advanced Motion Detection** with consecutive frame validation
- **Geographic Coordinate Conversion** using K-NN algorithm
- **Analytics Dashboard** with Chart.js visualizations
- **RESTful API** for external integrations
- **Scalable Architecture** with Celery and Redis

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Redis Server
- 4GB+ RAM

### Installation
```bash

cd car-tracking-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your settings

# Run migrations
python manage.py migrate

# Start services
redis-server  # In separate terminal
python manage.py runserver