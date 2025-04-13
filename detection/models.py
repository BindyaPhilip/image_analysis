from django.db import models
from django.utils import timezone
# Create your models here.

#create a model to store images and detection results
class RustDetectionResult(models.Model):
  

    image = models.ImageField(upload_to='uploads') #this needs further looking at
    uploaded_at = models.DateTimeField(default=timezone.now)
    rust_class = models.CharField(max_length=20, default='unknown')
    confidence = models.FloatField(null = True, blank=True) #this is the confidence score from the model


    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        confidence_str = f"{self.confidence:.2f}" if self.confidence is not None else "N/A"
        return f"Detection {self.id} - {'Rust' if self.rust_detected else 'No Rust'} (Confidence: {confidence_str})"
    

#model for storing training images and their labels
class TrainingImage(models.Model):
    #this will ensure only images with valid labels are uploaded
    CLASS_CHOICES = [
        ('common_rust', 'Common Rust'),
        ('healthy', 'Healthy'),
        ('other_disease', 'Other Disease')
    ]
    image = models.ImageField(upload_to='training_images')
    label = models.CharField(max_length=20, choices= CLASS_CHOICES)
    uploaded_at = models.DateTimeField(default=timezone.now)

    class Meta: #this class tells django how to handle the model
        ordering = ['-uploaded_at'] #ordering is used to specify the order in which dataset things are queried #the minus means the first uploaded things are queried first
    
    def __str__(self):
        return f"Training Image {self.id} - {self.label}"
