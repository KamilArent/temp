# Generated by Django 4.0 on 2024-05-20 15:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ai_app', '0006_airesponse'),
    ]

    operations = [
        migrations.AlterField(
            model_name='airesponse',
            name='response',
            field=models.ImageField(blank=True, null=True, upload_to='mediaphoto'),
        ),
    ]
