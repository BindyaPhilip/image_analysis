from django.urls import path
from .views import (
    RegisterFarmerView, RegisterExpertView, FarmerProfileView, ExpertProfileView,
    DiseaseHistoryView, CropHealthAlertView, CommunityPostView, CommunityPostModerationView,
    EducationalContentSubmissionView, AvailabilitySlotView, ConsultationBookingView,
    UserManagementView, SystemHealthView, ModelRetrainingView, FeedbackAnalysisView,
    ContentApprovalView,ExpertByIdView, FarmerByIdView,
    ChangePasswordView, ForgotPasswordView, ResetPasswordView
)
from . import views
urlpatterns = [
    path('register/farmer/', RegisterFarmerView.as_view(), name='register_farmer'),
    path('register/expert/', RegisterExpertView.as_view(), name='register_expert'),
    path('farmer/profile/', FarmerProfileView.as_view(), name='farmer_profile'),
    path('expert/profile/', ExpertProfileView.as_view(), name='expert_profile'),
    path('disease-history/', DiseaseHistoryView.as_view(), name='disease_history'),
    path('crop-health-alerts/', CropHealthAlertView.as_view(), name='crop_health_alerts'),
    path('community-posts/', CommunityPostView.as_view(), name='community_posts'),
    path('community-posts/moderate/<uuid:post_id>/', CommunityPostModerationView.as_view(), name='moderate_post'),
    path('educational-content/', EducationalContentSubmissionView.as_view(), name='educational_content'),
    path('availability-slots/', AvailabilitySlotView.as_view(), name='availability_slots'),
    path('consultations/', ConsultationBookingView.as_view(), name='consultations'),
    path('users/', UserManagementView.as_view(), name='user_management'),
    path('users/<uuid:user_id>/', UserManagementView.as_view(), name='user_management_detail'),
    path('system-health/', SystemHealthView.as_view(), name='system_health'),
    path('retrain-model/', ModelRetrainingView.as_view(), name='retrain_model'),
    path('feedback-analysis/', FeedbackAnalysisView.as_view(), name='feedback_analysis'),
    path('content-approval/', ContentApprovalView.as_view(), name='content_approval'),
    path('content-approval/<uuid:post_id>/', ContentApprovalView.as_view(), name='content_approval_detail'),
    path('crop-types/', views.CropTypeView.as_view(), name='crop_types'),
    path('reset-password/', ResetPasswordView.as_view(), name='reset_password'),
    path('forgot-password/', ForgotPasswordView.as_view(), name='forgot_password'),
    path('experts/<uuid:id>/', ExpertByIdView.as_view(), name='expert_by_id'),
    path('farmers/<uuid:id>/', FarmerByIdView.as_view(), name='farmer_by_id'),
    path('change-password/', ChangePasswordView.as_view(), name='change_password')
]