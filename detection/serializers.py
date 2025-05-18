from rest_framework import serializers
from .models import RustDetectionResult, TrainingImage

class RustDetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RustDetectionResult
        fields = ['id', 'farmer_id', 'image', 'uploaded_at', 'rust_class', 'confidence']
        read_only_fields = ['id', 'uploaded_at', 'rust_class', 'confidence', 'farmer_id']
        extra_kwargs = {
            'image': {'write_only': False, 'help_text': 'URL of the uploaded image'},
            'rust_class': {'help_text': 'Detected class (common_rust, healthy, other_disease)'},
            'confidence': {'help_text': 'Prediction confidence score (0-1)'},
        }

class DetectionItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = RustDetectionResult
        fields = ['id', 'image', 'uploaded_at', 'rust_class', 'confidence']
        read_only_fields = ['id', 'uploaded_at', 'rust_class', 'confidence']
        extra_kwargs = {
            'image': {'help_text': 'URL of the uploaded image'},
            'rust_class': {'help_text': 'Detected class (common_rust, healthy, other_disease)'},
            'confidence': {'help_text': 'Prediction confidence score (0-1)'},
        }

class DetectionHistorySerializer(serializers.Serializer):
    farmer_id = serializers.UUIDField(help_text='ID of the authenticated farmer')
    detections = DetectionItemSerializer(many=True, help_text='List of detection results')

class TrainingImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingImage
        fields = ['id', 'image', 'label', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']
        extra_kwargs = {
            'image': {'write_only': False, 'help_text': 'URL of the training image'},
            'label': {
                'help_text': 'Label for the image (common_rust, healthy, other_disease)',
                'choices': ['common_rust', 'healthy', 'other_disease']
            },
        }