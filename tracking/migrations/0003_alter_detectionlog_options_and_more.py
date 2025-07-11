# Generated by Django 4.2.20 on 2025-06-10 10:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("tracking", "0002_alter_detectionlog_session"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="detectionlog",
            options={"ordering": ["-timestamp"]},
        ),
        migrations.AlterField(
            model_name="detectionlog",
            name="session",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="detection_logs",
                to="tracking.cartrackingsession",
            ),
        ),
    ]
