from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .models import RustDetectionResult
from .serializers import RustDetectionResultSerializer, TrainingImageSerializer
import os
from django.conf import settings
import tensorflow as tf
import numpy as np
import requests
from django.core.wsgi import get_wsgi_application

class RustDetectionView(APIView):
    parser_classes = [MultiPartParser]

    MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'rust_model', 'multi_class_model.keras')
    print(f"Model path: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    def post(self, request, format=None):
        # Retrieve the uploaded image
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': "Image was not uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the image to RustDetectionResult
        detection = RustDetectionResult(image=image_file)
        detection.save()

        # Process the image for rust detection
        detection_result = self.detect_rust(detection.image.path)
        detection.rust_class = detection_result['rust_class']
        detection.confidence = detection_result['confidence']
        detection.save()

        # Fetch educational resources for the detected disease
        education_resources = self.get_education_resources(detection.rust_class)

        # Serialize the detection result
        serializer = RustDetectionResultSerializer(detection)
        response_data = serializer.data
        response_data['educational_resources'] = education_resources or []  # Include resources (empty list if none)
        return Response(response_data, status=status.HTTP_201_CREATED)
    
    def detect_rust(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)[0]
        class_names = ['common_rust', 'healthy', 'other_disease']
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[predicted_class_idx])
        return {
            'rust_class': predicted_class,
            'confidence': confidence
        }

    def get_education_resources(self, disease):
        """
        Fetch resources from the education microservice based on detected disease.
        """
        try:
            # Use the Ngrok URL of the education microservice
            education_service_url = 'https://d04f-160-119-149-222.ngrok-free.app/api/resources/'
            # Pass the detected disease as a query parameter
            params = {'disease': disease}
            response = requests.get(education_service_url, params=params, timeout=5)
            response.raise_for_status()  # Raise exception for 4xx/5xx errors
            return response.json()  # Return list of resources (e.g., [{"id": UUID, "title": ...}])
        except requests.RequestException as e:
            # Log error but return empty list to avoid breaking detection
            print(f"Failed to fetch resources: {e}")
            return []

class TrainingImageUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        serializer = TrainingImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)