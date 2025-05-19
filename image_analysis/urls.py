from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from django.http import JsonResponse

def root_view(request):
    return JsonResponse({
        "message": "Welcome to the Image Analysis API",
        "endpoints": {
            "api_docs": "/swagger/",
            "rust_detection": "/api/rust-detection/",
            "upload_training": "/api/upload-training-images/"
        }
    })


schema_view = get_schema_view(
    openapi.Info(
        title="Image Analysis API",
        default_version='v1',
        description="API for detecting crop rust diseases and uploading training images",
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="your-email@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('', root_view),  # Add this line
    path('admin/', admin.site.urls),
    path('api/', include('detection.urls')),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)