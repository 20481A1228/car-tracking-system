# Generated migration for car tracking models

from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CarTrackingSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('video_file', models.FileField(upload_to='videos/')),
                ('polygon_file', models.FileField(upload_to='polygons/')),
                ('geo_coords_file', models.FileField(blank=True, null=True, upload_to='geo_coords/')),
                ('reference_image', models.ImageField(blank=True, null=True, upload_to='reference_images/')),
                ('motion_threshold', models.IntegerField(default=30)),
                ('motion_sensitivity', models.FloatField(default=0.1)),
                ('min_motion_area', models.IntegerField(default=100)),
                ('consecutive_motion_frames', models.IntegerField(default=3)),
                ('k_neighbors', models.IntegerField(default=4)),
                ('location_precision', models.IntegerField(default=6)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('total_frames', models.IntegerField(default=0)),
                ('processed_frames', models.IntegerField(default=0)),
                ('cars_detected', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='Car',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('car_id', models.CharField(max_length=50)),
                ('first_detected', models.DateTimeField(auto_now_add=True)),
                ('last_seen', models.DateTimeField(auto_now=True)),
                ('total_detections', models.IntegerField(default=0)),
                ('is_active', models.BooleanField(default=True)),
                ('max_speed', models.FloatField(default=0.0)),
                ('avg_speed', models.FloatField(default=0.0)),
                ('total_distance', models.FloatField(default=0.0)),
                ('avg_motion_confidence', models.FloatField(default=0.0)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cars', to='tracking.cartrackingsession')),
            ],
        ),
        migrations.CreateModel(
            name='TrackingPoint',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField()),
                ('pixel_x', models.FloatField()),
                ('pixel_y', models.FloatField()),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('is_moving', models.BooleanField(default=True)),
                ('speed', models.FloatField(default=0.0)),
                ('motion_confidence', models.FloatField(default=0.0)),
                ('bbox_x1', models.FloatField()),
                ('bbox_y1', models.FloatField()),
                ('bbox_x2', models.FloatField()),
                ('bbox_y2', models.FloatField()),
                ('car', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tracking_points', to='tracking.car')),
            ],
            options={
                'ordering': ['timestamp'],
            },
        ),
        migrations.CreateModel(
            name='ReferencePoint',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pixel_x', models.FloatField()),
                ('pixel_y', models.FloatField()),
                ('yolo_x', models.FloatField()),
                ('yolo_y', models.FloatField()),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('point_order', models.IntegerField()),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='reference_points', to='tracking.cartrackingsession')),
            ],
            options={
                'ordering': ['point_order'],
            },
        ),
        migrations.CreateModel(
            name='DetectionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('message', models.TextField()),
                ('level', models.CharField(choices=[('INFO', 'Info'), ('WARNING', 'Warning'), ('ERROR', 'Error')], default='INFO', max_length=10)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='detection_logs', to='tracking.cartrackingsession')),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='car',
            unique_together={('session', 'car_id')},
        ),
    ]