import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage completely
import tensorflow as tf
import io
# Configure TensorFlow to use minimal resources
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)
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

class LeafDetectionView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    # Use relative path from Django's BASE_DIR
    LEAF_MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'rust_model', 'objects_splitting.keras')
    _leaf_model = None

    @classmethod
    def get_leaf_model(cls):
        """Load the leaf detection model with proper error handling"""
        if cls._leaf_model is None:
            try:
                # Verify the model file exists
                if not os.path.exists(cls.LEAF_MODEL_PATH):
                    raise FileNotFoundError(f"Leaf model file not found at: {cls.LEAF_MODEL_PATH}")
                
                # Clear session and configure TensorFlow
                tf.keras.backend.clear_session()
                tf.config.optimizer.set_jit(False)
                tf.config.optimizer.set_experimental_options({
                    "arithmetic_optimization": False,
                    "disable_meta_optimizer": True
                })
                
                # Load the model
                cls._leaf_model = tf.keras.models.load_model(cls.LEAF_MODEL_PATH)
                logger.info("Leaf detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load leaf detection model: {str(e)}")
                cls._leaf_model = None
                raise
        return cls._leaf_model

    @swagger_auto_schema(
        operation_description=(
            "Check if an uploaded image is a leaf.\n\n"
            "**Permissions**: Requires JWT authentication (farmer).\n"
            "**Form Data**:\n"
            "- `image`: Image file (JPEG/PNG).\n\n"
            "**Returns**: Classification result indicating if the image is a leaf."
        ),
        manual_parameters=[
            openapi.Parameter(
                name='image',
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description='Image file for leaf detection (JPEG/PNG)'
            ),
        ],
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'is_leaf': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='True if image is a leaf, False otherwise'),
                    'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format='float', description='Prediction confidence (0-1)')
                },
                example={
                    'is_leaf': True,
                    'confidence': 0.95
                }
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error message')
                },
                example={'error': 'Image was not uploaded'}
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Authentication error')
                },
                example={'detail': 'Authentication credentials were not provided'}
            ),
            500: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, description='Server error message')
                },
                example={'error': 'Leaf detection model not found'}
            )
        },
        security=[{'Bearer': []}]
    )
    def post(self, request, format=None):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': "Image was not uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # First verify the model exists
            if not os.path.exists(self.LEAF_MODEL_PATH):
                return Response(
                    {'error': f"Leaf detection model not found at: {self.LEAF_MODEL_PATH}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Read the InMemoryUploadedFile into a BytesIO object
            image_data = image_file.read()
            image_stream = io.BytesIO(image_data)

            # Process image for leaf detection (128x128 is the input size for the model)
            img = tf.keras.preprocessing.image.load_img(
                image_stream, 
                target_size=(128, 128),
                color_mode='rgb'
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Get the model and predict
            model = self.get_leaf_model()
            prediction = model.predict(img_array, verbose=0, batch_size=1)[0]

            # Clean up resources
            image_stream.close()
            del img, img_array
            tf.keras.backend.clear_session()

            # Assuming model outputs [leaf_prob, non_leaf_prob]
            is_leaf = prediction[0] > prediction[1]
            confidence = float(prediction[0] if is_leaf else prediction[1])

            return Response({
                'is_leaf': bool(is_leaf),  # Convert numpy bool to Python bool
                'confidence': confidence
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Leaf detection error: {str(e)}", exc_info=True)
            return Response(
                {'error': f"Leaf detection failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )










class RustDetectionView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]
    pagination_class = PageNumberPagination
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection', 'rust_model', 'multi_class_model.keras')
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            tf.keras.backend.clear_session()
            tf.config.optimizer.set_jit(False)
            tf.config.optimizer.set_experimental_options({
                "arithmetic_optimization": False,
                "disable_meta_optimizer": True
            })
            cls._model = tf.keras.models.load_model(cls.MODEL_PATH)
            logger.info("Model loaded successfully")
        return cls._model

    @swagger_auto_schema(
        operation_description=(
            "Upload an image for crop rust detection.\n\n"
            "**Permissions**: Requires JWT authentication (farmer).\n"
            "**Form Data**:\n"
            "- `image`: Image file (JPEG/PNG).\n\n"
            "**Returns**: Detection result with rust class, confidence, feedback, and educational resources."
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
                    'feedback': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'message': openapi.Schema(type=openapi.TYPE_STRING, description='Farmer-friendly message'),
                            'explanation': openapi.Schema(type=openapi.TYPE_STRING, description='Condition explanation'),
                            'advice': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Schema(type=openapi.TYPE_STRING), description='Actionable advice'),
                            'confidence': openapi.Schema(type=openapi.TYPE_STRING, description='Confidence level description')
                        }
                    ),
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
                    'feedback': {
                        'message': 'Your plant appears to have common rust.',
                        'explanation': 'Common rust is a fungal disease often seen as orange or yellow pustules on leaves.',
                        'advice': ['Apply a fungicide like azoxystrobin.', 'Remove affected leaves.', 'Ensure good air circulation.'],
                        'confidence': 'We are highly confident (score: 0.95).'
                    },
                    'educational_resources': [{'id': 1, 'title': 'Managing Common Rust', 'content': 'Details...'}]
                }
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING, description='Error message')}
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'detail': openapi.Schema(type=openapi.TYPE_STRING, description='Authentication error')}
            )
        },
        security=[{'Bearer': []}]
    )
    def post(self, request, format=None):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': "Image was not uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        farmer_id = request.user.id if hasattr(request.user, 'id') else None
        if not farmer_id:
            logger.warning("No farmer_id found in JWT token")
            return Response({'error': "Authentication does not provide user ID"}, status=status.HTTP_400_BAD_REQUEST)

        # Perform leaf detection
        try:
            leaf_response = LeafDetectionView().post(request)
            if leaf_response.status_code != 200 or not leaf_response.data['is_leaf']:
                return Response({
                    'error': 'Uploaded image is not a leaf',
                    'confidence': leaf_response.data.get('confidence', 0)
                }, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Leaf detection check failed: {str(e)}")
            return Response({'error': f"Leaf detection check failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save detection result
        detection = RustDetectionResult(image=image_file, farmer_id=farmer_id)
        detection.save()

        # Perform rust detection
        detection_result = self.detect_rust(detection.image.path)
        detection.rust_class = detection_result['rust_class']
        detection.confidence = detection_result['confidence']
        detection.save()

        # Generate farmer-friendly feedback
        confidence_level = (
            "highly confident" if detection.confidence > 0.80 else
            "moderately confident" if detection.confidence > 0.50 else
            "not very confident, so please verify with an expert"
        )
        if detection.confidence < 0.50:
            feedback = {
                'message': "The image is unclear, or the plant's condition is uncertain.",
                'explanation': "We couldn't confidently identify the plant's condition.",
                'advice': [
                    "Upload a clearer, well-lit image of the plant leaves.",
                    "Consult a local agricultural expert for a detailed diagnosis."
                ],
                'confidence': f"We are {confidence_level} (score: {detection.confidence:.2f})."
            }
        elif detection.rust_class == 'healthy':
            feedback = {
                'message': "Good news! Your plant looks healthy.",
                'explanation': "The plant shows no signs of disease based on the image.",
                'advice': [
                    "Continue regular care, such as proper watering and fertilization.",
                    "Monitor for pests or environmental stress to maintain health."
                ],
                'confidence': f"We are {confidence_level} (score: {detection.confidence:.2f})."
            }
        elif detection.rust_class == 'common_rust':
            feedback = {
                'message': "Your plant appears to have common rust.",
                'explanation': "Common rust is a fungal disease often seen as orange or yellow pustules on leaves.",
                'advice': [
                    "Apply a fungicide like azoxystrobin, following local guidelines.",
                    "Remove affected leaves to reduce spread.",
                    "Ensure good air circulation by spacing plants properly."
                ],
                'confidence': f"We are {confidence_level} (score: {detection.confidence:.2f})."
            }
        else:  # other_disease
            feedback = {
                'message': "Your plant may have a disease, but it’s not clearly common rust.",
                'explanation': "The image shows signs of a potential disease, possibly fungal, bacterial, or viral.",
                'advice': [
                    "Consult a local agricultural extension service for a detailed diagnosis.",
                    "Check for symptoms like leaf spots, wilting, or discoloration.",
                    "Avoid watering from above to prevent worsening fungal issues."
                ],
                'confidence': f"We are {confidence_level} (score: {detection.confidence:.2f})."
            }

        # Fetch educational resources
        education_resources = self.get_education_resources(detection.rust_class)

        # Serialize response
        serializer = RustDetectionResultSerializer(detection)
        response_data = serializer.data
        response_data['feedback'] = feedback
        response_data['educational_resources'] = education_resources or []
        response_data['message'] = f"No resources found for {detection.rust_class}" if not education_resources else None
        logger.info(f"Detection saved for farmer {farmer_id}: {detection.rust_class}")
        return Response(response_data, status=status.HTTP_201_CREATED)

    # get and detect_rust methods remain unchanged
    def get(self, request, format=None):
        farmer_id = request.user.id if hasattr(request.user, 'id') else None
        if not farmer_id:
            logger.warning("No farmer_id found in JWT token for GET request")
            return Response({'error': "Authentication does not provide user ID"}, status=status.HTTP_400_BAD_REQUEST)
        detections = RustDetectionResult.objects.filter(farmer_id=farmer_id)
        paginator = self.pagination_class()
        page = paginator.paginate_queryset(detections, request)
        item_serializer = DetectionItemSerializer(page, many=True)
        data = {'farmer_id': str(farmer_id), 'detections': item_serializer.data}
        history_serializer = DetectionHistorySerializer(data)
        return paginator.get_paginated_response(history_serializer.data)

    def detect_rust(self, image_path):
        try:
            model = self.get_model()
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array, verbose=0, batch_size=1)[0]
            del img, img_array
            tf.keras.backend.clear_session()
            class_names = ['common_rust', 'healthy', 'other_disease']
            return {
                'rust_class': class_names[np.argmax(prediction)],
                'confidence': float(prediction[np.argmax(prediction)])
            }
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            raise

    def get_education_resources(self, disease):
        try:
            education_service_url = 'http://localhost:8000/api/resources/'
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