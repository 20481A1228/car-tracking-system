# setup_directories.py - Run this script to create all required directories

import os
from pathlib import Path

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent

def create_directories():
    """Create all required directories for the car tracking system."""
    
    directories = [
        # Media directories for file uploads
        'media',
        'media/videos',
        'media/polygons', 
        'media/geo_coords',
        'media/reference_images',
        
        # Static files directory
        'static',
        'static/css',
        'static/js',
        'static/images',
        
        # Templates directory (if not exists)
        'templates',
        
        # Logs directory
        'logs',
    ]
    
    print("🚗 Setting up Car Tracking System directories...")
    print("="*50)
    
    for directory in directories:
        dir_path = BASE_DIR / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if dir_path.exists():
            print(f"✅ Created: {directory}/")
        else:
            print(f"❌ Failed: {directory}/")
    
    print("="*50)
    print("✅ Directory setup complete!")
    
    # Create a .gitkeep file in media directories to ensure they're tracked by git
    media_dirs = ['media/videos', 'media/polygons', 'media/geo_coords', 'media/reference_images']
    for media_dir in media_dirs:
        gitkeep_file = BASE_DIR / media_dir / '.gitkeep'
        gitkeep_file.touch()
    
    print("📁 File structure:")
    print_directory_tree()

def print_directory_tree():
    """Print the directory structure."""
    tree = """
    car_tracking_project/
    ├── manage.py
    ├── car_tracking_project/
    │   ├── settings.py ✅ (updated)
    │   ├── urls.py ✅ (updated)
    │   └── wsgi.py
    ├── media/ ✅ (created)
    │   ├── videos/ ✅
    │   ├── polygons/ ✅
    │   ├── geo_coords/ ✅
    │   └── reference_images/ ✅
    ├── static/ ✅ (created)
    ├── templates/ ✅ (created)
    ├── logs/ ✅ (created)
    ├── tracking/
    │   ├── models.py
    │   ├── views.py
    │   ├── forms.py
    │   └── urls.py
    └── api/
        ├── views.py
        ├── serializers.py
        └── urls.py
    """
    print(tree)

if __name__ == "__main__":
    create_directories()

# ==================================================
# QUICK DIAGNOSTIC COMMANDS
# ==================================================

def check_setup():
    """Check if everything is set up correctly."""
    print("\n🔍 CHECKING SETUP...")
    print("="*30)
    
    # Check directories
    required_dirs = ['media', 'templates', 'static', 'logs']
    for directory in required_dirs:
        path = BASE_DIR / directory
        status = "✅ EXISTS" if path.exists() else "❌ MISSING"
        print(f"{directory}/: {status}")
    
    # Check files
    required_files = [
        'car_tracking_project/settings.py',
        'car_tracking_project/urls.py',
        'tracking/models.py',
        'tracking/views.py',
        'tracking/forms.py',
        'api/views.py',
        'api/serializers.py'
    ]
    
    print("\n📄 CHECKING FILES...")
    for file_path in required_files:
        path = BASE_DIR / file_path
        status = "✅ EXISTS" if path.exists() else "❌ MISSING"
        print(f"{file_path}: {status}")

if __name__ == "__main__":
    create_directories()
    check_setup()

print("\n🎯 NEXT STEPS:")
print("1. Replace your settings.py with the fixed version above")
print("2. Replace your main urls.py with the fixed version above") 
print("3. Run: python manage.py migrate")
print("4. Run: python manage.py runserver")
print("5. Test creating a session at: http://localhost:8000/create/")