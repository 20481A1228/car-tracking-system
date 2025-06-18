from django.test import TestCase

# Create your tests here.
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from .models import CarTrackingSession, Car, TrackingPoint
from .coordinate_converter import CoordinateConverter
import json

class CoordinateConverterTestCase(TestCase):
    def setUp(self):
        # Sample YOLO coordinates for testing
        self.yolo_coords = [
            (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)
        ]
        # Sample geo coordinates
        self.geo_coords = [
            (40.4584, -79.9284), (40.4584, -79.9280),
            (40.4580, -79.9280), (40.4580, -79.9284)
        ]
        
        self.converter = CoordinateConverter(
            image_path=None,
            NBlot_coords=self.yolo_coords,
            geo_coords=self.geo_coords,
            image_width=640,
            image_height=360
        )

    def test_yolo_to_pixel_conversion(self):
        pixel_coords = self.converter.yolo_to_pixel_coords(self.yolo_coords)
        expected = [(64, 36), (576, 36), (576, 324), (64, 324)]
        self.assertEqual(pixel_coords, expected)

    def test_point_in_polygon(self):
        # Test point inside polygon
        inside_point = (320, 180)  # Center of image
        self.assertTrue(self.converter.point_in_polygon(inside_point, self.converter.NBlot_pixels))
        
        # Test point outside polygon
        outside_point = (10, 10)
        self.assertFalse(self.converter.point_in_polygon(outside_point, self.converter.NBlot_pixels))

    def test_geolocation_conversion(self):
        # Test conversion for center point
        lat, lon = self.converter.compute_geolocation_knn(320, 180)
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lon)
        self.assertAlmostEqual(lat, 40.4582, places=4)
        self.assertAlmostEqual(lon, -79.9282, places=4)

class CarTrackingSessionTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        
        # Create test files
        self.video_file = SimpleUploadedFile(
            "test_video.mp4", b"fake video content", content_type="video/mp4"
        )
        self.polygon_file = SimpleUploadedFile(
            "test_polygon.txt", b"0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9", content_type="text/plain"
        )
        
        self.session_data = {
            'name': 'Test Session',
            'motion_threshold': 30,
            'motion_sensitivity': 0.1,
            'min_motion_area': 100,
            'consecutive_motion_frames': 3,
            'k_neighbors': 4,
            'location_precision': 6,
        }

    def test_create_session(self):
        session = CarTrackingSession.objects.create(
            name='Test Session',
            video_file=self.video_file,
            polygon_file=self.polygon_file
        )
        self.assertEqual(session.name, 'Test Session')
        self.assertEqual(session.status, 'pending')

    def test_dashboard_view(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dashboard')

    def test_create_session_view(self):
        response = self.client.get(reverse('create_session'))
        self.assertEqual(response.status_code, 200)

class CarTrackingAPITestCase(TestCase):
    def setUp(self):
        self.session = CarTrackingSession.objects.create(
            name='API Test Session',
            video_file=SimpleUploadedFile("test.mp4", b"content"),
            polygon_file=SimpleUploadedFile("test.txt", b"0 0.1 0.1")
        )
        
        self.car = Car.objects.create(
            session=self.session,
            car_id='test-car-1',
            total_detections=10,
            max_speed=25.5
        )
        
        TrackingPoint.objects.create(
            car=self.car,
            timestamp='2023-01-01 12:00:00',
            pixel_x=320,
            pixel_y=180,
            latitude=40.4584,
            longitude=-79.9284,
            speed=15.0,
            motion_confidence=0.8,
            bbox_x1=300, bbox_y1=160,
            bbox_x2=340, bbox_y2=200
        )

    def test_api_sessions_list(self):
        response = self.client.get('/api/sessions/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(len(data['results']), 1)

    def test_api_session_stats(self):
        response = self.client.get(f'/api/sessions/{self.session.id}/stats/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['total_cars'], 1)

    def test_api_car_path(self):
        response = self.client.get(f'/api/cars/{self.car.id}/path/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(len(data), 1)

class MotionDetectionTestCase(TestCase):
    def test_calculate_iou(self):
        from .car_detector import CarDetector
        
        # Mock session
        session = CarTrackingSession.objects.create(
            name='IoU Test',
            video_file=SimpleUploadedFile("test.mp4", b"content"),
            polygon_file=SimpleUploadedFile("test.txt", b"0 0.1 0.1")
        )
        
        detector = CarDetector(session.id)
        
        # Test overlapping boxes
        box1 = (100, 100, 200, 200)
        box2 = (150, 150, 250, 250)
        iou = detector.calculate_iou(box1, box2)
        
        # Expected IoU for these boxes is 1/7 ≈ 0.143
        self.assertAlmostEqual(iou, 0.143, places=2)
        
        # Test identical boxes
        iou_identical = detector.calculate_iou(box1, box1)
        self.assertEqual(iou_identical, 1.0)
        
        # Test non-overlapping boxes
        box3 = (300, 300, 400, 400)
        iou_no_overlap = detector.calculate_iou(box1, box3)
        self.assertEqual(iou_no_overlap, 0.0)

class PerformanceTestCase(TestCase):
    def test_coordinate_conversion_performance(self):
        import time
        
        # Large number of reference points
        yolo_coords = [(i*0.01, j*0.01) for i in range(10) for j in range(10)]
        geo_coords = [(40.4584 + i*0.0001, -79.9284 + j*0.0001) for i in range(10) for j in range(10)]
        
        converter = CoordinateConverter(
            image_path=None,
            NBlot_coords=yolo_coords,
            geo_coords=geo_coords
        )
        
        # Test performance of 1000 conversions
        start_time = time.time()
        for i in range(1000):
            lat, lon = converter.compute_geolocation_knn(320 + i%100, 180 + i%100)
        end_time = time.time()
        
        # Should complete 1000 conversions in less than 1 second
        self.assertLess(end_time - start_time, 1.0)

# Run tests with: python manage.py test