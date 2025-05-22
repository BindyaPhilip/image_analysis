from django.urls import path
from .views import RustDetectionView,TrainingImageUploadView,LeafDetectionView

urlpatterns = [
    path('rust-detection/', RustDetectionView.as_view(), name='rust-detection'),
    path('leaf-detection/', LeafDetectionView.as_view(), name='leaf-detection'),
    path('upload-training-images/',TrainingImageUploadView.as_view(), name='upload-training-images' )
]