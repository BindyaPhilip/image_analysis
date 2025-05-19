import os
import logging
import numpy as np
import tensorflow as tf
import requests
from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.pagination import PageNumberPagination
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .models import RustDetectionResult
from .serializers import RustDetectionResultSerializer, TrainingImageSerializer, DetectionHistorySerializer, DetectionItemSerializer
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
logger = logging.getLogger(__name__)
# Reduce TensorFlow logging
tf.get_logger().setLevel('ERROR')
# Limit GPU memory growth if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class RustDetectionView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    pagination_class = PageNumberPagination  # Define pagination_class

    MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'rust_model', 'multi_class_model.keras')
    _model = None  # Private class variable for lazy loading
    
    @property
    def model(self):
        """Lazy-load the model on first access"""
        if self._model is None:
            # Configure TensorFlow to be more memory efficient
            tf.keras.backend.clear_session()  # Clear any existing sessions
            self._model = tf.keras.models.load_model(self.MODEL_PATH)
            print("Model loaded successfully")
        return self._model
    

    @swagger_auto_schema(
        operation_description=(
            "Upload an image for crop rust detection.\n\n"
            "**Permissions**: Requires JWT authentication (farmer).\n"
            "**Form Data**:\n"
            "- `image`: Image file (JPEG/PNG).\n\n"
            "**Returns**: Detection result with rust class, confidence, and educational resources."
        ),
        manual_parameters=[
            openapi.Parameter(
                name='image',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description='Image file for rust detection (JPEG/PNG)'
            ),
        ],
        responses={
            201: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'id': openapi.Schema(type=openapi.TYPE_INTEGER, description='Detection ID'),
                    'farmer_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                    'image': openapi.Schema(type=openapi.TYPE_STRING, description='Image URL'),
                    'uploaded_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', description='Upload timestamp'),
                    'rust_class': openapi.Schema(type=openapi.TYPE_STRING, description='Detected class (common_rust, healthy, other_disease)'),
                    'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format='float', description='Prediction confidence (0-1)'),
                    'educational_resources': openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_OBJECT),
                        description='Educational resources from Education microservice'
                    ),
                    'message': openapi.Schema(type=openapi.TYPE_STRING, description='Message if no resources found', nullable=True),
                },
                example={
                    'id': 1,
                    'farmer_id': '123e4567-e89b-12d3-a456-426614174000',
                    'image': '/media/uploads/image.jpg',
                    'uploaded_at': '2025-05-03T12:00:00Z',
                    'rust_class': 'common_rust',
                    'confidence': 0.95,
                    'educational_resources': [
                        {'id': 1, 'title': 'Managing Common Rust', 'content': 'Details...'},
                    ]
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error message')
                },
                example={'error': 'Image was not uploaded'},
                description='Invalid request (e.g., missing image or invalid farmer ID)'
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Authentication error')
                },
                example={'detail': 'Authentication credentials were not provided'},
                description='Unauthorized access'
            ),
        },
        security=[{'Bearer': []}],
    )
    def post(self, request, format=None):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': "Image was not uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        farmer_id = request.user.id if hasattr(request.user, 'id') else None
        if not farmer_id:
            logger.warning("No farmer_id found in JWT token")
            return Response({'error': "Authentication does not provide user ID"}, status=status.HTTP_400_BAD_REQUEST)

        detection = RustDetectionResult(image=image_file, farmer_id=farmer_id)
        detection.save()

        detection_result = self.detect_rust(detection.image.path)
        detection.rust_class = detection_result['rust_class']
        detection.confidence = detection_result['confidence']
        detection.save()

        education_resources = self.get_education_resources(detection.rust_class)

        serializer = RustDetectionResultSerializer(detection)
        response_data = serializer.data
        response_data['educational_resources'] = education_resources or []
        response_data['message'] = f"No resources found for {detection.rust_class}" if not education_resources else None
        logger.info(f"Detection saved for farmer {farmer_id}: {detection.rust_class}")
        return Response(response_data, status=status.HTTP_201_CREATED)
    
    @swagger_auto_schema(
        operation_description=(
            "Retrieve paginated rust detection history for the authenticated farmer.\n\n"
            "**Permissions**: Requires JWT authentication (farmer).\n"
            "**Query Parameters**:\n"
            "- `page`: Page number (default: 1).\n"
            "- `page_size`: Number of results per page (default: 10).\n\n"
            "**Returns**: Paginated list of past detection results with farmer ID."
        ),
        manual_parameters=[
            openapi.Parameter(
                name='page',
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_INTEGER,
                required=False,
                description='Page number (default: 1)'
            ),
            openapi.Parameter(
                name='page_size',
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_INTEGER,
                required=False,
                description='Number of results per page (default: 10)'
            ),
        ],
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'count': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total number of detections'),
                    'next': openapi.Schema(type=openapi.TYPE_STRING, description='URL to next page', nullable=True),
                    'previous': openapi.Schema(type=openapi.TYPE_STRING, description='URL to previous page', nullable=True),
                    'results': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'farmer_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                            'detections': openapi.Schema(
                                type=openapi.TYPE_ARRAY,
                                items=openapi.Schema(
                                    type=openapi.TYPE_OBJECT,
                                    properties={
                                        'id': openapi.Schema(type=openapi.TYPE_INTEGER, description='Detection ID'),
                                        'image': openapi.Schema(type=openapi.TYPE_STRING, description='Image URL'),
                                        'uploaded_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time', description='Upload timestamp'),
                                        'rust_class': openapi.Schema(type=openapi.TYPE_STRING, description='Detected class (common_rust, healthy, other_disease)'),
                                        'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format='float', description='Prediction confidence (0-1)'),
                                    },
                                ),
                            ),
                        },
                    ),
                },
                example={
                    'count': 2,
                    'next': 5,
                    'previous': 4,
                    'results': {
                        'farmer_id': '123e4567-e89b-12d3-a456-426614174000',
                        'detections': [
                            {
                                'id': 40,
                                'image': '/media/uploads/maize_rust_fqWMpZD.jpg',
                                'uploaded_at': '2025-05-06T19:44:52.517909Z',
                                'rust_class': 'other_disease',
                                'confidence': 0.9259061813354492,
                            },
                            {
                                'id': 37,
                                'image': '/media/uploads/Corn_Common_Rust_9_XL8oVd6.jpg',
                                'uploaded_at': '2025-05-03T16:36:52.941115Z',
                                'rust_class': 'common_rust',
                                'confidence': 0.988950788974762,
                            },
                        ],
                    },
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error message')
                },
                example={'error': 'Authentication does not provide user ID'},
                description='Invalid farmer ID'
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Authentication error')
                },
                example={'detail': 'Authentication credentials were not provided'},
                description='Unauthorized access'
            ),
        },
        security=[{'Bearer': []}],
    )
    def get(self, request, format=None):
        farmer_id = request.user.id if hasattr(request.user, 'id') else None
        if not farmer_id:
            logger.warning("No farmer_id found in JWT token for GET request")
            return Response({'error': "Authentication does not provide user ID"}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch detections and paginate
        detections = RustDetectionResult.objects.filter(farmer_id=farmer_id)
        paginator = self.pagination_class()
        page = paginator.paginate_queryset(detections, request)
        
        # Serialize detections without farmer_id
        item_serializer = DetectionItemSerializer(page, many=True)
        # Create response with farmer_id and detections
        data = {
            'farmer_id': str(farmer_id),
            'detections': item_serializer.data
        }
        history_serializer = DetectionHistorySerializer(data)
        
        # Return paginated response
        return paginator.get_paginated_response(history_serializer.data)

    def detect_rust(self, image_path):
        try:
            # Load and process image
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict and get results
            prediction = self.model.predict(img_array, verbose=0)[0]  # verbose=0 silences output
            class_names = ['common_rust', 'healthy', 'other_disease']
            predicted_class_idx = np.argmax(prediction)
            
            # Explicit cleanup
            del img, img_array
            tf.keras.backend.clear_session()
            
            return {
                'rust_class': class_names[predicted_class_idx],
                'confidence': float(prediction[predicted_class_idx])
            }
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            raise

    def get_education_resources(self, disease):
        try:
            education_service_url = 'http://localhost:8001/api/resources/'
            params = {'disease': disease}
            response = requests.get(education_service_url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch resources: {e}")
            return []

class TrainingImageUploadView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_description=(
            "Upload a labeled training image for model improvement.\n\n"
            "**Permissions**: Requires JWT authentication (farmer).\n"
            "**Form Data**:\n"
            "- `image`: Image file (JPEG/PNG).\n"
            "- `label`: Image label (common_rust, healthy, other_disease).\n\n"
            "**Returns**: Details of the uploaded training image."
        ),
        manual_parameters=[
            openapi.Parameter(
                name='image',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description='Image file for training (JPEG/PNG)'
            ),
            openapi.Parameter(
                name='label',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                enum=['common_rust', 'healthy', 'other_disease'],
                description='Label for the training image'
            ),
        ],
        responses={
            201: TrainingImageSerializer,
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'image': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING)),
                    'label': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING)),
                },
                example={
                    'image': ['This field is required'],
                    'label': ['This field is required']
                },
                description='Validation errors (e.g., missing image or invalid label)'
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Authentication error')
                },
                example={'detail': 'Authentication credentials were not provided'},
                description='Unauthorized access'
            ),
        },
        security=[{'Bearer': []}],
    )
    def post(self, request, format=None):
        serializer = TrainingImageSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            logger.info(f"Training image uploaded: {serializer.data['label']}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)