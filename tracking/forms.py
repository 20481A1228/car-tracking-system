from django import forms
from .models import CarTrackingSession

class CarTrackingSessionForm(forms.ModelForm):
    """Form for creating and editing car tracking sessions."""
    
    class Meta:
        model = CarTrackingSession
        fields = ['name', 'video_file', 'polygon_file', 'geo_coords_file', 
                 'reference_image', 'motion_threshold', 'motion_sensitivity',
                 'min_motion_area', 'consecutive_motion_frames', 'k_neighbors',
                 'location_precision']
        
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter session name'
            }),
            'video_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*'
            }),
            'polygon_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.txt'
            }),
            'geo_coords_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.txt'
            }),
            'reference_image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'motion_threshold': forms.NumberInput(attrs={
                'class': 'form-control', 
                'min': 10, 
                'max': 100,
                'value': 30
            }),
            'motion_sensitivity': forms.NumberInput(attrs={
                'class': 'form-control', 
                'step': 0.01, 
                'min': 0.01, 
                'max': 1.0,
                'value': 0.1
            }),
            'min_motion_area': forms.NumberInput(attrs={
                'class': 'form-control', 
                'min': 50, 
                'max': 1000,
                'value': 100
            }),
            'consecutive_motion_frames': forms.NumberInput(attrs={
                'class': 'form-control', 
                'min': 1, 
                'max': 10,
                'value': 3
            }),
            'k_neighbors': forms.NumberInput(attrs={
                'class': 'form-control', 
                'min': 1, 
                'max': 20,
                'value': 4
            }),
            'location_precision': forms.NumberInput(attrs={
                'class': 'form-control', 
                'min': 1, 
                'max': 10,
                'value': 6
            }),
        }
        
        labels = {
            'name': 'Session Name',
            'video_file': 'Video File',
            'polygon_file': 'Polygon Boundary File (YOLO format)',
            'geo_coords_file': 'Geographic Coordinates File (Optional)',
            'reference_image': 'Reference Image (Optional)',
            'motion_threshold': 'Motion Threshold (10-100)',
            'motion_sensitivity': 'Motion Sensitivity (0.01-1.0)',
            'min_motion_area': 'Minimum Motion Area (50-1000)',
            'consecutive_motion_frames': 'Consecutive Motion Frames (1-10)',
            'k_neighbors': 'K-Nearest Neighbors (1-20)',
            'location_precision': 'Location Precision (1-10 decimal places)',
        }
        
        help_texts = {
            'name': 'A descriptive name for this tracking session',
            'video_file': 'Video file to analyze for car tracking',
            'polygon_file': 'YOLO format file defining the tracking area boundary',
            'geo_coords_file': 'CSV file with lat,lng coordinates (optional)',
            'reference_image': 'Reference image for coordinate mapping (optional)',
            'motion_threshold': 'Threshold for motion detection sensitivity',
            'motion_sensitivity': 'Percentage of car area that must be in motion',
            'min_motion_area': 'Minimum motion area to consider valid movement',
            'consecutive_motion_frames': 'Number of consecutive frames needed for motion',
            'k_neighbors': 'Number of nearest reference points for coordinate conversion',
            'location_precision': 'Number of decimal places for coordinate precision',
        }

    def clean_video_file(self):
        """Validate video file."""
        video = self.cleaned_data.get('video_file')
        if video:
            # Check file size (limit to 500MB)
            if video.size > 500 * 1024 * 1024:
                raise forms.ValidationError('Video file too large. Maximum size is 500MB.')
            
            # Check file extension
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            if not any(video.name.lower().endswith(ext) for ext in valid_extensions):
                raise forms.ValidationError('Invalid video format. Supported formats: MP4, AVI, MOV, MKV, WMV')
        
        return video

    def clean_polygon_file(self):
        """Validate polygon file."""
        polygon = self.cleaned_data.get('polygon_file')
        if polygon:
            if not polygon.name.lower().endswith('.txt'):
                raise forms.ValidationError('Polygon file must be a .txt file.')
            
            # Basic content validation
            try:
                content = polygon.read().decode('utf-8')
                polygon.seek(0)  # Reset file pointer
                
                lines = content.strip().split('\n')
                if not lines:
                    raise forms.ValidationError('Polygon file appears to be empty.')
                
                # Check if it looks like YOLO format (class + coordinates)
                first_line = lines[0].strip().split()
                if len(first_line) < 7:  # At least class + 3 coordinate pairs
                    raise forms.ValidationError('Polygon file does not appear to be in YOLO format.')
                
            except UnicodeDecodeError:
                raise forms.ValidationError('Polygon file must be a text file.')
        
        return polygon

    def clean_geo_coords_file(self):
        """Validate geo coordinates file."""
        geo_file = self.cleaned_data.get('geo_coords_file')
        if geo_file:
            valid_extensions = ['.csv', '.txt']
            if not any(geo_file.name.lower().endswith(ext) for ext in valid_extensions):
                raise forms.ValidationError('Geo coordinates file must be .csv or .txt format.')
        
        return geo_file

    def clean_motion_threshold(self):
        """Validate motion threshold."""
        threshold = self.cleaned_data.get('motion_threshold')
        if threshold is not None and (threshold < 10 or threshold > 100):
            raise forms.ValidationError('Motion threshold must be between 10 and 100.')
        return threshold

    def clean_motion_sensitivity(self):
        """Validate motion sensitivity."""
        sensitivity = self.cleaned_data.get('motion_sensitivity')
        if sensitivity is not None and (sensitivity < 0.01 or sensitivity > 1.0):
            raise forms.ValidationError('Motion sensitivity must be between 0.01 and 1.0.')
        return sensitivity

print("✅ Tracking Forms file created!")
print("📁 Save this as: tracking/forms.py")
print("🎯 This provides the form for creating tracking sessions!")