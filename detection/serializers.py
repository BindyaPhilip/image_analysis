from rest_framework import serializers
from .models import RustDetectionResult, TrainingImage

#this serializer bridges the gap between the model and the api by converting things into json format

class RustDetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RustDetectionResult
        fields = ['id', 'image', 'uploaded_at', 'rust_class', 'confidence']
        read_only_fields = ['id', 'uploaded_at', 'rust_class', 'confidence']

class TrainingImageSerializer(serializers.ModelSerializer):
    class Meta: 
        model = TrainingImage
        fields = ['id', 'image', 'label', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']
