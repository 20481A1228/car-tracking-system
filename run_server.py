#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "car_tracking_project.settings")
    
    from daphne.cli import CommandLineInterface
    
    # Run with Daphne
    sys.argv = [
        "daphne",
        "-b", "127.0.0.1", 
        "-p", "8000",
        "car_tracking_project.asgi:application"
    ]
    
    CommandLineInterface.entrypoint()