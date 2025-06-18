#!/usr/bin/env python
import os
import sys

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "car_tracking_project.settings")
    
    # Force ASGI for development
    if len(sys.argv) > 1 and sys.argv[1] == 'runserver':
        from daphne.management.commands.runserver import Command as DaphneCommand
        from django.core.management import execute_from_command_line
        from django.conf import settings
        
        # Use Daphne for runserver
        if settings.DEBUG:
            os.system('daphne -b 127.0.0.1 -p 8000 car_tracking_project.asgi:application')
            return
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    main()