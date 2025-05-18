import os
import sys
import logging
import numpy as np
import tensorflow as tf
from django.core.management.base import BaseCommand
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from detection.models import TrainingImage

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'retrain_model.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Command(BaseCommand):
    help = 'Retrain the rust detection model with uploaded training images'

    def handle(self, *args, **options):
        logging.info("Starting model retraining process")
        self.stdout.write("Starting model retraining process")

        # Paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_PATH = os.path.join(BASE_DIR, 'detection', 'rust_model', 'multi_class_model.keras')
        logging.info(f"Model path: {MODEL_PATH}")

        # Load existing model
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logging.info(f"Loaded model from {MODEL_PATH}")
            self.stdout.write(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            self.stderr.write(f"Failed to load model: {str(e)}")
            return

        # Data generator for augmentation
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        logging.info("Initialized data generator")

        # Load and augment data
        images, labels = self.load_training_data()
        if len(images) == 0:
            logging.warning("No training images available. Skipping retraining.")
            self.stdout.write("No training images available. Skipping retraining.")
            return

        # Retrain the model
        try:
            model.fit(
                images,
                labels,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            logging.info("Model retraining completed")
            self.stdout.write("Model retraining completed")
        except Exception as e:
            logging.error(f"Error during model retraining: {str(e)}")
            self.stderr.write(f"Error during model retraining: {str(e)}")
            return

        # Save the updated model
        try:
            model.save(MODEL_PATH)
            logging.info(f"Updated model saved to {MODEL_PATH}")
            self.stdout.write(f"Updated model saved to {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            self.stderr.write(f"Failed to save model: {str(e)}")

    def load_training_data(self):
        logging.info("Loading training data")
        try:
            images = []
            labels = []
            class_names = ['common_rust', 'healthy', 'other_disease']
            for img_obj in TrainingImage.objects.all():
                img_path = img_obj.image.path
                logging.info(f"Processing image: {img_path}")
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    images.append(img_array)
                    label = tf.keras.utils.to_categorical(class_names.index(img_obj.label), num_classes=3)
                    labels.append(label)
                except Exception as e:
                    logging.error(f"Failed to process image {img_path}: {str(e)}")
                    continue
            logging.info(f"Loaded {len(images)} images")
            return np.array(images), np.array(labels)
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            return np.array([]), np.array([])