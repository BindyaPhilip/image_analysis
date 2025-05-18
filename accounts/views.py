# Import APIView to create class-based views for handling HTTP requests
import uuid
from rest_framework.views import APIView
# Import Response to return JSON responses to clients
from rest_framework.response import Response
# Import status to use HTTP status codes (e.g., 201 Created, 400 Bad Request)
from rest_framework import status
# Import PageNumberPagination for paginating large query results
from rest_framework.pagination import PageNumberPagination

from user_management import settings
# Import models to interact with database objects
from .models import User, FarmerProfile, ExpertProfile, CropType, AvailabilitySlot, ConsultationBooking, CommunityPost, Feedback, SystemMetric,PasswordResetToken, RustAlert
# Import serializers to validate input and serialize output to JSON
from .serializers import ChangePasswordSerializer, UserSerializer, FarmerProfileSerializer, ExpertProfileSerializer, AvailabilitySlotSerializer, ConsultationBookingSerializer, CommunityPostSerializer, FeedbackSerializer, SystemMetricSerializer,RegisterFarmerSerializer,ForgotPasswordSerializer,ResetPasswordSerializer

# Import custom permissions to restrict access based on user roles
from .permissions import IsFarmer, IsExpert, IsAdmin, IsFarmerOrAdmin, IsExpertOrAdmin
# Import AllowAny to allow unauthenticated access (e.g., for registration)
from rest_framework.permissions import AllowAny
# Import requests to make HTTP calls to other microservices
import requests
# Import send_mail to send email alerts
from django.core.mail import send_mail
# Import shared_task to define Celery tasks for asynchronous operations
from celery import shared_task
# Import timezone for timezone-aware date/time handling
from django.utils import timezone
# Import timedelta for time-based calculations (e.g., recent feedback)
from datetime import timedelta
# Import swagger_auto_schema for Swagger documentation
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import logging

logger = logging.getLogger(__name__)

# Celery task to send asynchronous email alerts to farmers about detected diseases
@shared_task
def send_alert_email(farmer_email, disease):
    """
    Celery task to send email alerts about detected diseases.
    
    Args:
        farmer_email (str): Email address of the farmer.
        disease (str): Name of the detected disease.
    """
    subject = f"Disease Alert: {disease} Detected"
    message = f"Multiple detections of {disease} have been recorded on your farm. Please consult an expert or refer to educational resources."
    send_mail(subject, message, 'from@example.com', [farmer_email])

class CropTypeView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'value': openapi.Schema(type=openapi.TYPE_STRING, description='Crop Type Value'),
                        'label': openapi.Schema(type=openapi.TYPE_STRING, description='Crop Type Label'),
                    },
                ),
            ),
        },
        security=[]
    )
    def get(self, request):
        crop_types = [{'value': value, 'label': label} for value, label in CropType.choices()]
        return Response(crop_types)






# API view to handle farmer registration
class RegisterFarmerView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'email': openapi.Schema(type=openapi.TYPE_STRING, format='email'),
                'username': openapi.Schema(type=openapi.TYPE_STRING),
                'password': openapi.Schema(type=openapi.TYPE_STRING, format='password'),
                'confirm_password': openapi.Schema(type=openapi.TYPE_STRING, format='password'),
                'phone_number': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'farm_location': openapi.Schema(type=openapi.TYPE_STRING),
                'farm_size': openapi.Schema(type=openapi.TYPE_NUMBER, format='float'),
                'crop_types': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_STRING, enum=[ct.value for ct in CropType]),
                    nullable=True
                ),
            },
            required=['email', 'username', 'password', 'confirm_password', 'farm_location', 'farm_size'],
        ),
        responses={
            201: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'user': openapi.Schema(type=openapi.TYPE_OBJECT),
                    'profile': openapi.Schema(type=openapi.TYPE_OBJECT),
                },
            ),
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
        },
        security=[]
    )
    def post(self, request):
        serializer = RegisterFarmerSerializer(data=request.data)
        if serializer.is_valid():
            result = serializer.create(serializer.validated_data)
            return Response(result, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# API view to handle expert registration
class RegisterExpertView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'email': openapi.Schema(type=openapi.TYPE_STRING, format='email'),
                'username': openapi.Schema(type=openapi.TYPE_STRING),
                'password': openapi.Schema(type=openapi.TYPE_STRING, format='password'),
                'phone_number': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'areas_of_expertise': openapi.Schema(type=openapi.TYPE_STRING),
                'certifications': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'bio': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'experience_years': openapi.Schema(type=openapi.TYPE_INTEGER, nullable=True),
                'institution': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'languages_spoken': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                'social_links': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
            },
            required=['email', 'username', 'password', 'areas_of_expertise'],
        ),
        responses={
            201: openapi.Schema(type=openapi.TYPE_OBJECT),
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
        },
        security=[]
    )
    def post(self, request):
        data = request.data.copy()
        data['role'] = 'expert'
        data['is_active'] = True
        data['is_approved'] = True
        user_serializer = UserSerializer(data=data)
        if user_serializer.is_valid():
            user = user_serializer.save()
            profile_data = {
                'areas_of_expertise': data.get('areas_of_expertise', ''),
                'certifications': data.get('certifications', ''),
                'bio': data.get('bio', ''),
                'experience_years': data.get('experience_years', 0),
                'institution': data.get('institution', ''),
                'languages_spoken': data.get('languages_spoken', ''),
                'social_links': data.get('social_links', ''),
            }
            profile_serializer = ExpertProfileSerializer(data=profile_data)
            if profile_serializer.is_valid():
                # Save the profile with the user instance
                profile = ExpertProfile.objects.create(user=user, **profile_serializer.validated_data)
                return Response(
                    {
                        'user': UserSerializer(user).data,
                        'profile': ExpertProfileSerializer(profile).data
                    },
                    status=status.HTTP_201_CREATED
                )
            user.delete()
            return Response(profile_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(user_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#API for farmer profile and update
class FarmerProfileView(APIView):
    permission_classes = [IsFarmer]

    @swagger_auto_schema(
        responses={200: FarmerProfileSerializer},
        security=[{'Bearer': []}]
    )
    def get(self, request):
        profile = request.user.farmer_profile
        serializer = FarmerProfileSerializer(profile)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=FarmerProfileSerializer,
        responses={200: FarmerProfileSerializer, 400: openapi.Schema(type=openapi.TYPE_OBJECT)},
        security=[{'Bearer': []}]
    )
    def put(self, request):
        profile = request.user.farmer_profile
        serializer = FarmerProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# API view for experts to view or update their profile
class ExpertProfileView(APIView):
    permission_classes = [IsExpert]

    @swagger_auto_schema(
        responses={200: ExpertProfileSerializer},
        security=[{'Bearer': []}]
    )
    def get(self, request):
        profile = request.user.expert_profile
        serializer = ExpertProfileSerializer(profile)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=ExpertProfileSerializer,
        responses={
            200: ExpertProfileSerializer,
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
            403: openapi.Schema(type=openapi.TYPE_OBJECT)
        },
        security=[{'Bearer': []}]
    )
    def put(self, request):
        profile = request.user.expert_profile
        serializer = ExpertProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# API view for farmers to view disease detection history from image_analysis microservice
class DiseaseHistoryView(APIView):
    permission_classes = [IsFarmer]
    pagination_class = PageNumberPagination

    @swagger_auto_schema(
        responses={200: openapi.Schema(type=openapi.TYPE_OBJECT), 400: openapi.Schema(type=openapi.TYPE_OBJECT)},
        security=[{'Bearer': []}]
    )
    def get(self, request):
        try:
            response = requests.get('http://localhost:8000/api/rust-detection/', headers={'Authorization': f'Bearer {request.auth}'})
            response.raise_for_status()
            detections = response.json()
            paginator = self.pagination_class()
            page = paginator.paginate_queryset(detections, request)
            return paginator.get_paginated_response(page)
        except requests.RequestException as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


# API view for farmers to check crop health alerts based on detection frequency
class CropHealthAlertView(APIView):
    permission_classes = [IsFarmer]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'status': openapi.Schema(type=openapi.TYPE_STRING),
                    'common_rust_count': openapi.Schema(type=openapi.TYPE_INTEGER),
                    'alert_sent': openapi.Schema(type=openapi.TYPE_BOOLEAN)
                }
            ),
            400: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        try:
            # Fetch farmer-specific detections from image_analysis microservice
            response = requests.get(
                'http://localhost:8000/api/rust-detection/',
                headers={'Authorization': f'Bearer {request.auth}'},
                timeout=5
            )
            response.raise_for_status()
            detections = response.json()

            # Validate detections format
            if not isinstance(detections, list):
                logger.error("Invalid detection data format")
                return Response({'error': 'Invalid detection data'}, status=status.HTTP_400_BAD_REQUEST)

            # Count common rust detections
            common_rust_count = sum(1 for d in detections if d.get('rust_class') == 'common_rust')
            alert_sent = False

            # Check if alert should be sent (count > 5 and no recent alert)
            if common_rust_count > 5:
                recent_alert = RustAlert.objects.filter(
                    farmer=request.user,
                    sent_at__gte=timezone.now() - timedelta(hours=24)
                ).exists()
                if not recent_alert:
                    # Send email to farmer
                    subject = 'Crop Health Alert: Common Rust Detected'
                    message = (
                        f"Dear {request.user.username},\n\n"
                        f"We have detected common rust on your farm {common_rust_count} times.\n"
                        f"Please take immediate action to address this issue.\n"
                        f"Contact an agricultural expert via our platform if needed.\n"
                    )
                    try:
                        send_mail(
                            subject=subject,
                            message=message,
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[request.user.email],
                            fail_silently=False,
                        )
                        # Record the alert
                        RustAlert.objects.create(
                            farmer=request.user,
                            detection_count=common_rust_count
                        )
                        alert_sent = True
                        logger.info(f"Sent common rust alert to {request.user.email}: {common_rust_count} detections")
                    except Exception as e:
                        logger.error(f"Failed to send alert to {request.user.email}: {str(e)}")
                        return Response({'error': f'Failed to send email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                'status': 'Checked for alerts',
                'common_rust_count': common_rust_count,
                'alert_sent': alert_sent
            }, status=status.HTTP_200_OK)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch detections: {str(e)}")
            return Response({'error': f'Failed to fetch detection data: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
# API view for farmers to view/post community posts, admins to view
class CommunityPostView(APIView):
    permission_classes = [IsFarmerOrAdmin]

    @swagger_auto_schema(
        responses={200: CommunityPostSerializer(many=True)},
        security=[{'Bearer': []}]
    )
    def get(self, request):
        posts = CommunityPost.objects.filter(is_approved=True)
        serializer = CommunityPostSerializer(posts, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=CommunityPostSerializer,
        responses={201: CommunityPostSerializer, 400: openapi.Schema(type=openapi.TYPE_OBJECT), 403: openapi.Schema(type=openapi.TYPE_OBJECT)},
        security=[{'Bearer': []}]
    )
    def post(self, request):
        if request.user.role != 'farmer':
            return Response({'error': 'Only farmers can post'}, status=status.HTTP_403_FORBIDDEN)
        data = request.data.copy()
        data['farmer'] = request.user.id
        serializer = CommunityPostSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# API view for experts/admins to moderate community posts
class CommunityPostModerationView(APIView):
    """
    List unapproved community posts or moderate a specific post.
    
    **Permissions**: Requires authenticated expert or admin role.
    
    **GET /api/community-posts/moderate/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with list of unapproved posts.
    
    **PUT /api/community-posts/moderate/<uuid:post_id>/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with action (approve, flag, unflag).
    - Response: 200 OK with updated post, 400 Bad Request for invalid action, or 404 Not Found if post doesn’t exist.
    """
    permission_classes = [IsExpertOrAdmin]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Post ID'),
                        'title': openapi.Schema(type=openapi.TYPE_STRING),
                        'content': openapi.Schema(type=openapi.TYPE_STRING),
                        'farmer': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                        'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'is_flagged': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                    },
                ),
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        posts = CommunityPost.objects.filter(is_approved=False)
        serializer = CommunityPostSerializer(posts, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'action': openapi.Schema(type=openapi.TYPE_STRING, enum=['approve', 'flag', 'unflag'], description='Action to perform'),
            },
            required=['action'],
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Post ID'),
                    'title': openapi.Schema(type=openapi.TYPE_STRING),
                    'content': openapi.Schema(type=openapi.TYPE_STRING),
                    'farmer': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                    'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    'is_flagged': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def put(self, request, post_id):
        try:
            post = CommunityPost.objects.get(id=post_id)
            action = request.data.get('action')
            if action == 'approve':
                post.is_approved = True
            elif action == 'flag':
                post.is_flagged = True
            elif action == 'unflag':
                post.is_flagged = False
            else:
                return Response({'error': 'Invalid action'}, status=status.HTTP_400_BAD_REQUEST)
            post.save()
            return Response(CommunityPostSerializer(post).data)
        except CommunityPost.DoesNotExist:
            return Response({'error': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)

# API view for experts to submit educational content to education microservice
class EducationalContentSubmissionView(APIView):
    """
    Submit educational content to the education microservice.
    
    **Permissions**: Requires authenticated expert role.
    
    **POST /api/educational-content/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with content details (title, url, description, disease, resource_type).
    - Response: 201 Created with microservice response, or 400 Bad Request if microservice fails.
    """
    permission_classes = [IsExpert]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'title': openapi.Schema(type=openapi.TYPE_STRING, description='Content title'),
                'url': openapi.Schema(type=openapi.TYPE_STRING, format='uri', description='Content URL'),
                'description': openapi.Schema(type=openapi.TYPE_STRING, description='Content description'),
                'disease': openapi.Schema(type=openapi.TYPE_STRING, description='Related disease'),
                'resource_type': openapi.Schema(type=openapi.TYPE_STRING, description='Resource type (e.g., Article, Video)'),
            },
            required=['title', 'url', 'description', 'disease', 'resource_type'],
        ),
        responses={
            201: openapi.Schema(type=openapi.TYPE_OBJECT, description='Response from education microservice'),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def post(self, request):
        data = {
            'title': request.data.get('title'),
            'url': request.data.get('url'),
            'description': request.data.get('description'),
            'disease': request.data.get('disease'),
            'resource_type': request.data.get('resource_type'),
        }
        try:
            response = requests.post('http://localhost:8001/api/resources/', json=data, headers={'Authorization': f'Bearer {request.auth}'})
            response.raise_for_status()
            return Response(response.json(), status=status.HTTP_201_CREATED)
        except requests.RequestException as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# API view for experts to manage availability slots
class AvailabilitySlotView(APIView):
    """
    List or create availability slots for consultations.
    
    **Permissions**: Requires authenticated expert role.
    
    **GET /api/availability-slots/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with list of unbooked slots.
    
    **POST /api/availability-slots/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with slot details (start_time, end_time).
    - Response: 201 Created with slot data, or 400 Bad Request if validation fails.
    """
    permission_classes = [IsExpert]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Slot ID'),
                        'expert': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Expert ID'),
                        'start_time': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                        'end_time': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                        'is_booked': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    },
                ),
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        slots = AvailabilitySlot.objects.filter(expert=request.user, is_booked=False)
        serializer = AvailabilitySlotSerializer(slots, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=AvailabilitySlotSerializer,
        responses={
            201: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Slot ID'),
                    'expert': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Expert ID'),
                    'start_time': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                    'end_time': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                    'is_booked': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def post(self, request):
        data = request.data.copy()
        data['expert'] = request.user.id
        serializer = AvailabilitySlotSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# API view for farmers to manage consultation bookings
class ConsultationBookingView(APIView):
    """
    List or create consultation bookings.
    
    **Permissions**: Requires authenticated farmer role.
    
    **GET /api/consultations/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with list of bookings.
    
    **POST /api/consultations/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with slot_id.
    - Response: 201 Created with booking data, or 400 Bad Request if slot is unavailable.
    """
    permission_classes = [IsFarmer]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Booking ID'),
                        'farmer': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                        'slot': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Slot ID'),
                        'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                    },
                ),
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        bookings = ConsultationBooking.objects.filter(farmer=request.user)
        serializer = ConsultationBookingSerializer(bookings, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'slot_id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='ID of the availability slot'),
            },
            required=['slot_id'],
        ),
        responses={
            201: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Booking ID'),
                    'farmer': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Farmer ID'),
                    'slot': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Slot ID'),
                    'created_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def post(self, request):
        slot_id = request.data.get('slot_id')
        try:
            slot = AvailabilitySlot.objects.get(id=slot_id, is_booked=False)
            booking = ConsultationBooking(farmer=request.user, slot=slot)
            slot.is_booked = True
            slot.save()
            booking.save()
            serializer = ConsultationBookingSerializer(booking)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except AvailabilitySlot.DoesNotExist:
            return Response({'error': 'Slot unavailable'}, status=status.HTTP_400_BAD_REQUEST)

# API view for admins to manage users
class UserManagementView(APIView):
    """
    List all users or update a specific user.
    
    **Permissions**: Requires authenticated admin role.
    
    **GET /api/users/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with list of users.
    
    **PUT /api/users/<uuid:user_id>/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with action (approve, block) or updated user fields.
    - Response: 200 OK with updated user, 400 Bad Request for invalid data, or 404 Not Found if user doesn’t exist.
    """
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='User ID'),
                        'email': openapi.Schema(type=openapi.TYPE_STRING, format='email'),
                        'username': openapi.Schema(type=openapi.TYPE_STRING),
                        'role': openapi.Schema(type=openapi.TYPE_STRING, enum=['farmer', 'expert', 'admin']),
                        'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'is_active': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                        'phone_number': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                    },
                ),
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'action': openapi.Schema(type=openapi.TYPE_STRING, enum=['approve', 'block'], description='Action to perform (optional)'),
                'email': openapi.Schema(type=openapi.TYPE_STRING, format='email', description='User email (optional)'),
                'username': openapi.Schema(type=openapi.TYPE_STRING, description='Username (optional)'),
                'role': openapi.Schema(type=openapi.TYPE_STRING, enum=['farmer', 'expert', 'admin'], description='User role (optional)'),
                'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Approval status (optional)'),
                'is_active': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Active status (optional)'),
            },
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='User ID'),
                    'email': openapi.Schema(type=openapi.TYPE_STRING, format='email'),
                    'username': openapi.Schema(type=openapi.TYPE_STRING),
                    'role': openapi.Schema(type=openapi.TYPE_STRING, enum=['farmer', 'expert', 'admin']),
                    'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    'is_active': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    'phone_number': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
            404: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def put(self, request, user_id):
        try:
            user = User.objects.get(id=user_id)
            action = request.data.get('action')
            if action == 'approve':
                user.is_approved = True
            elif action == 'block':
                user.is_approved = False
                user.is_active = False
            else:
                serializer = UserSerializer(user, data=request.data, partial=True)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            user.save()
            return Response(UserSerializer(user).data)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

# API view for admins to view system metrics
class SystemHealthView(APIView):
    """
    View system metrics.
    
    **Permissions**: Requires authenticated admin role.
    
    **GET /api/system-health/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with list of system metrics.
    """
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Items(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'id': openapi.Schema(type=openapi.TYPE_STRING, format='uuid', description='Metric ID'),
                        'metric_name': openapi.Schema(type=openapi.TYPE_STRING),
                        'metric_value': openapi.Schema(type=openapi.TYPE_NUMBER, format='float'),
                        'recorded_at': openapi.Schema(type=openapi.TYPE_STRING, format='date-time'),
                    },
                ),
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        metrics = SystemMetric.objects.all()
        serializer = SystemMetricSerializer(metrics, many=True)
        return Response(serializer.data)

# API view for admins to trigger model retraining
class ModelRetrainingView(APIView):
    """
    Trigger model retraining in the image_analysis microservice.
    
    **Permissions**: Requires authenticated admin role.
    
    **POST /api/retrain-model/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Request Body: JSON with training data (passed to microservice).
    - Response: 200 OK if retraining triggered, or 400 Bad Request if microservice fails.
    """
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            description='Training data (format depends on image_analysis microservice)',
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'status': openapi.Schema(type=openapi.TYPE_STRING, description='Retraining status')},
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={'error': openapi.Schema(type=openapi.TYPE_STRING)},
            ),
        },
        security=[{'Bearer': []}]
    )
    def post(self, request):
        try:
            response = requests.post('http://localhost:8000/api/upload-training-images/', data=request.data, headers={'Authorization': f'Bearer {request.auth}'})
            response.raise_for_status()
            return Response({'status': 'Retraining triggered'}, status=status.HTTP_200_OK)
        except requests.RequestException as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

# API view for admins to analyze feedback
class FeedbackAnalysisView(APIView):
    """
    Analyze feedback summary.
    
    **Permissions**: Requires authenticated admin role.
    
    **GET /api/feedback-analysis/**
    - Headers: Authorization: Bearer <JWT_TOKEN>
    - Response: 200 OK with feedback summary (total and recent feedback counts).
    """
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'total_feedback': openapi.Schema(type=openapi.TYPE_INTEGER, description='Total feedback count'),
                    'recent_feedback': openapi.Schema(type=openapi.TYPE_INTEGER, description='Feedback count in last 30 days'),
                },
            ),
        },
        security=[{'Bearer': []}]
    )
    def get(self, request):
        feedback = Feedback.objects.all()
        total = feedback.count()
        recent = feedback.filter(created_at__gte=timezone.now() - timedelta(days=30)).count()
        return Response({'total_feedback': total, 'recent_feedback': recent})

# API view for admins to approve community posts
class ContentApprovalView(APIView):
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={200: CommunityPostSerializer(many=True)},
        security=[{'Bearer': []}]
    )
    def get(self, request):
        posts = CommunityPost.objects.filter(is_approved=False)
        serializer = CommunityPostSerializer(posts, many=True)
        return Response(serializer.data)

    @swagger_auto_schema(
        request_body=openapi.Schema(type=openapi.TYPE_OBJECT, properties={'is_approved': openapi.Schema(type=openapi.TYPE_BOOLEAN)}),
        responses={200: CommunityPostSerializer, 404: openapi.Schema(type=openapi.TYPE_OBJECT)},
        security=[{'Bearer': []}]
    )
    def put(self, request, post_id):
        try:
            post = CommunityPost.objects.get(id=post_id)
            post.is_approved = request.data.get('is_approved', False)
            post.save()
            return Response(CommunityPostSerializer(post).data)
        except CommunityPost.DoesNotExist:
            return Response({'error': 'Post not found'}, status=status.HTTP_404_NOT_FOUND)


class ResetPasswordView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ResetPasswordSerializer,
        responses={
            200: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'message': openapi.Schema(type=openapi.TYPE_STRING)}),
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
            404: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})
        },
        security=[]
    )
    def post(self, request):
        serializer = ResetPasswordSerializer(data=request.data)
        if serializer.is_valid():
            try:
                reset_token = PasswordResetToken.objects.get(token=serializer.validated_data['token'])
                if not reset_token.is_valid():
                    reset_token.delete()
                    return Response({'error': 'Token has expired'}, status=status.HTTP_400_BAD_REQUEST)
                user = reset_token.user
                user.set_password(serializer.validated_data['new_password'])
                user.save()
                reset_token.delete()
                return Response({'message': 'Password reset successfully'}, status=status.HTTP_200_OK)
            except PasswordResetToken.DoesNotExist:
                return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class ForgotPasswordView(APIView):
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        request_body=ForgotPasswordSerializer,
        responses={
            200: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'message': openapi.Schema(type=openapi.TYPE_STRING)}),
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
        },
        security=[]
    )
    def post(self, request):
        serializer = ForgotPasswordSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            user = User.objects.get(email=email)
            token = str(uuid.uuid4())
            expires_at = timezone.now() + timedelta(hours=1)

            # Store reset token
            PasswordResetToken.objects.create(
                user=user,
                token=token,
                expires_at=expires_at
            )

            # Send email using django.core.mail
            reset_url = f"{settings.FRONTEND_URL}/reset-password?token={token}"
            subject = 'Password Reset Request'
            message = (
                f"Hi {user.username},\n\n"
                f"You requested a password reset. Click the link below to set a new password:\n"
                f"{reset_url}\n\n"
                f"This link will expire in 1 hour.\n"
                f"If you didn't request this, please ignore this email.\n"
            )
            try:
                send_mail(
                    subject=subject,
                    message=message,
                    from_email=settings.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    fail_silently=False,
                )
            except Exception as e:
                return Response({'error': f'Failed to send email: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({'message': 'Password reset email sent'})
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


class ChangePasswordView(APIView):
    permission_classes = [IsExpert, IsFarmer]

    @swagger_auto_schema(
        request_body=ChangePasswordSerializer,
        responses={
            200: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'message': openapi.Schema(type=openapi.TYPE_STRING)}),
            400: openapi.Schema(type=openapi.TYPE_OBJECT),
            403: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})
        },
        security=[{'Bearer': []}]
    )
    def put(self, request):
        serializer = ChangePasswordSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            user = request.user
            user.set_password(serializer.validated_data['new_password'])
            user.save()
            return Response({'message': 'Password changed successfully'}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class FarmerByIdView(APIView):
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={
            200: FarmerProfileSerializer,
            404: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)}),
            403: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})
        },
        security=[{'Bearer': []}]
    )
    def get(self, request, id):
        try:
            farmer_profile = FarmerProfile.objects.get(user__id=id, user__role='farmer')
            serializer = FarmerProfileSerializer(farmer_profile)
            return Response(serializer.data)
        except FarmerProfile.DoesNotExist:
            return Response({'error': 'Farmer not found'}, status=status.HTTP_404_NOT_FOUND)
        
class ExpertByIdView(APIView):
    permission_classes = [IsAdmin]

    @swagger_auto_schema(
        responses={
            200: ExpertProfileSerializer,
            404: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)}),
            403: openapi.Schema(type=openapi.TYPE_OBJECT, properties={'error': openapi.Schema(type=openapi.TYPE_STRING)})
        },
        security=[{'Bearer': []}]
    )
    def get(self, request, id):
        try:
            expert_profile = ExpertProfile.objects.get(user__id=id, user__role='expert')
            serializer = ExpertProfileSerializer(expert_profile)
            return Response(serializer.data)
        except ExpertProfile.DoesNotExist:
            return Response({'error': 'Expert not found'}, status=status.HTTP_404_NOT_FOUND)