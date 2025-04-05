from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .models import RustDetectionResult
from .serializers import RustDetectionResultSerializer  # Fixed typo
import os
from django.conf import settings
import tensorflow as tf
import numpy as np
import requests


#the APIView is the base class for creating api endpoints
class RustDetectionView(APIView):
    #this handles file uploads
    parser_classes = [MultiPartParser]

    #loading the model after class initialization
    MODEL_PATH = os.path.join(settings.BASE_DIR,'detection','rust_model','multi_class_model.keras')
    model = tf.keras.models.load_model(MODEL_PATH)

    #handling 
    def post(self, request, format=None):
        #retrieve the uploaded image
        image_file = request.FILES.get('image')
        #handle a situation where the image is not present
        if not image_file:
            return Response({'error':"Image was not uploaded"},status=status.HTTP_400_BAD_REQUEST)
        #save the image to the database(RustDetectionResult)
        detection = RustDetectionResult(image=image_file)
        #the image is saved to the location where we specified in the IMAGEFIELD when defining the RustDetectionResult model 
        detection.save()

        #processing the image---get the result by calling the method detect_rust
        detection_result = self.detect_rust(detection.image.path)
        #update the detection record with resultse
        #syntax understanding.....detection.confidence- refers to the field in the model...detection_result['confidence']-this extracts from the dictionary returned by the method
        detection.rust_detected = detection_result['rust_detected']
        detection.confidence = detection_result['confidence']
        #save the updated model instance to the database
        detection.save()

        #HANDLING THE EDUCATION RESOURCES
        #initialize the variable that will handle the education resources
        education_resources = None
        if detection.rust_detected:
            education_resources = self.get_education_resources()

        #serialize the response
        serializer = RustDetectionResultSerializer(detection)
        response_data = serializer.data
        if education_resources:
            response_data['education_resources'] = education_resources
        return Response(response_data,status=status.HTTP_201_CREATED)
    

    def detect_rust(self,image_path):
        #load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array/255.0 #normalizing

        #predicting using the loaded model
        prediction = self.model.predict(img_array)[0]  ## Probabilities for [common_rust, healthy, other_disease]
        class_names = ['common_rust','healthy','other_disease']
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction[predicted_class_idx])

        #determine if it is common rust
        rust_detected = (predicted_class == 'common_rust')

        return {
            'rust_detected': rust_detected,
            'confidence': confidence
        }

    #STILL NEEDS MORE HANDLING
    def get_education_resources(self):
        """Communicate with the education resources microservice."""
        try:
            # Assuming the education microservice is a REST API
            education_service_url = 'http://education-service/api/resources/rust'  # Replace with actual URL
            response = requests.get(education_service_url, timeout=5)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()  # Expected to return JSON with resources
        except requests.RequestException as e:
            # Log the error (in production, use logging instead of print)
            print(f"Failed to fetch education resources: {e}")
            return {'error': 'Unable to fetch education resources'}
