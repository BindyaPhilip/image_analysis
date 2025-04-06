from django.urls import path
from .views import RustDetectionView

urlpatterns = [
    path('rust-detection/', RustDetectionView.as_view(), name='rust-detection'),
]