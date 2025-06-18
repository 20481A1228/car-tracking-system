from celery import shared_task
from .car_detector import CarDetector
from .models import CarTrackingSession

@shared_task
def process_video_async(session_id):
    try:
        session = CarTrackingSession.objects.get(id=session_id)
        detector = CarDetector(session_id)
        detector.process_video()
        return f"Successfully processed session {session.name}"
    except Exception as e:
        return f"Error processing session: {str(e)}"