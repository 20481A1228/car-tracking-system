from django.core.management.base import BaseCommand
from tracking.models import CarTrackingSession
from tracking.car_detector import CarDetector

class Command(BaseCommand):
    help = 'Process video for car tracking'

    def add_arguments(self, parser):
        parser.add_argument('session_id', type=int, help='Session ID to process')

    def handle(self, *args, **options):
        session_id = options['session_id']
        
        try:
            session = CarTrackingSession.objects.get(id=session_id)
            self.stdout.write(f'Starting processing for session: {session.name}')
            
            detector = CarDetector(session_id)
            detector.process_video()
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully processed session {session.name}')
            )
        except CarTrackingSession.DoesNotExist:
            self.stdout.write(
                self.style.ERROR(f'Session with ID {session_id} does not exist')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error processing session: {str(e)}')
            )