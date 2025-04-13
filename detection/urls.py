from django.urls import path
from .views import RustDetectionView,TrainingImageUploadView

urlpatterns = [
    path('rust-detection/', RustDetectionView.as_view(), name='rust-detection'),
    path('upload-training-images/',TrainingImageUploadView.as_view(), name='upload-training-images' )
]