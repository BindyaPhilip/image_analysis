from django.db import models
from django.utils import timezone
# Create your models here.

#create a model to store images and detection results

class RustDetectionResult(models.Model):
  

    image = models.ImageField(upload_to='/uploads') #this needs further looking at
    uploaded_at = models.DateTimeField(default=timezone.now)
    rust_detected = models.BooleanField(default=False)
    confidence = models.FloatField(null = True, blank=True) #this is the confidence score from the model


    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"Detection {self.id} - {'Rust' if self.rust_detected else 'No Rust'} (Confidence: {self.confidence})"
