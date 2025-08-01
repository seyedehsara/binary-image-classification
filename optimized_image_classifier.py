#!/usr/bin/env python3
"""
Optimized Image Classifier for Happy/Sad Detection
Performance-optimized version of the notebook with significant improvements.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import time
import argparse


class OptimizedImageClassifier:
    def __init__(self, data_dir='data', model_dir='models', log_dir='logs'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.image_classes = ['happy', 'sad']
        self.model = None
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure GPU memory growth
        self._configure_gpu()
        
        # Enable mixed precision for performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def _configure_gpu(self):
        """Configure GPU settings for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    
    def clean_dataset(self):
        """Remove corrupted images from dataset"""
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        removed_count = 0
        
        for image_class in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, image_class)
            if not os.path.isdir(class_path):
                continue
                
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Removing corrupted image: {image_path}")
                        os.remove(image_path)
                        removed_count += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    os.remove(image_path)
                    removed_count += 1
        
        print(f"Removed {removed_count} corrupted images")
    
    def load_and_preprocess_data(self, image_size=(224, 224), batch_size=32):
        """Load and preprocess data with performance optimizations"""
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Load dataset with optimizations
        data = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=42
        )
        
        # Preprocessing functions
        def preprocess_data(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        def augment_data(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.image.random_saturation(image, 0.9, 1.1)
            return image, label
        
        # Apply preprocessing
        data = data.map(preprocess_data, num_parallel_calls=AUTOTUNE)
        data = data.cache().prefetch(AUTOTUNE)
        
        # Split data
        dataset_size = len(data)
        train_size = int(dataset_size * 0.7)
        val_size = int(dataset_size * 0.2)
        
        train_data = data.take(train_size)
        train_data = train_data.map(augment_data, num_parallel_calls=AUTOTUNE)
        train_data = train_data.shuffle(1000).prefetch(AUTOTUNE)
        
        val_data = data.skip(train_size).take(val_size)
        val_data = val_data.cache().prefetch(AUTOTUNE)
        
        test_data = data.skip(train_size + val_size)
        test_data = test_data.cache().prefetch(AUTOTUNE)
        
        return train_data, val_data, test_data
    
    def build_optimized_model(self, input_shape=(224, 224, 3)):
        """Build an optimized CNN model with reduced parameters"""
        model = Sequential([
            # First block
            Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Second block
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Third block
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Classification head - highly optimized
            GlobalAveragePooling2D(),  # Reduces parameters by ~95%
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid', dtype='float32')  # Ensure float32 output for mixed precision
        ])
        
        # Optimized compilation
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(
            optimizer=optimizer,
            loss=tf.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self):
        """Get optimized training callbacks"""
        callbacks = [
            TensorBoard(self.log_dir, histogram_freq=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
            ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        return callbacks
    
    def train(self, train_data, val_data, epochs=50):
        """Train the model with all optimizations"""
        if self.model is None:
            raise ValueError("Model not built. Call build_optimized_model() first.")
        
        callbacks = self.get_callbacks()
        
        print("Starting optimized training...")
        start_time = time.time()
        
        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not available. Train or load a model first.")
        
        results = self.model.evaluate(test_data, verbose=0)
        print(f"Test Accuracy: {results[1]:.4f}")
        return results
    
    def optimize_for_deployment(self):
        """Create optimized models for deployment"""
        if self.model is None:
            raise ValueError("Model not available. Train or load a model first.")
        
        # Save in SavedModel format first
        saved_model_path = os.path.join(self.model_dir, 'saved_model')
        self.model.save(saved_model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        tflite_path = os.path.join(self.model_dir, 'optimized_model.tflite')
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model sizes
        original_size = os.path.getsize(os.path.join(self.model_dir, 'best_model.h5')) / (1024*1024)
        tflite_size = len(tflite_model) / (1024*1024)
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"TensorFlow Lite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
        
        return tflite_path
    
    def predict(self, image_path):
        """Optimized prediction for single image"""
        if self.model is None:
            raise ValueError("Model not available. Train or load a model first.")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        
        # Predict
        prediction = self.model.predict(img, verbose=0)
        predicted_class = self.image_classes[round(prediction[0][0])]
        confidence = prediction[0][0] if predicted_class == 'sad' else 1 - prediction[0][0]
        
        return predicted_class, confidence
    
    def benchmark_performance(self, test_data, num_runs=10):
        """Benchmark model inference performance"""
        if self.model is None:
            raise ValueError("Model not available. Train or load a model first.")
        
        times = []
        for batch in test_data.take(num_runs):
            start_time = time.time()
            _ = self.model.predict(batch[0], verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"Average inference time per batch: {avg_time:.4f} seconds")
        return avg_time


def main():
    parser = argparse.ArgumentParser(description="Optimized Image Classifier")
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--model_dir', default='models', help='Model directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--predict', help='Path to image for prediction')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = OptimizedImageClassifier(args.data_dir, args.model_dir)
    
    if args.predict:
        # Load existing model and predict
        model_path = os.path.join(args.model_dir, 'best_model.h5')
        if os.path.exists(model_path):
            classifier.model = tf.keras.models.load_model(model_path)
            result, confidence = classifier.predict(args.predict)
            print(f"Prediction: {result} (confidence: {confidence:.2f})")
        else:
            print("No trained model found. Please train first.")
    else:
        # Full training pipeline
        print("Starting optimized training pipeline...")
        
        # Clean dataset
        classifier.clean_dataset()
        
        # Load and preprocess data
        train_data, val_data, test_data = classifier.load_and_preprocess_data(
            batch_size=args.batch_size
        )
        
        # Build model
        model = classifier.build_optimized_model()
        print(f"Model built with {model.count_params():,} parameters")
        
        # Train model
        history = classifier.train(train_data, val_data, epochs=args.epochs)
        
        # Evaluate model
        classifier.evaluate_model(test_data)
        
        # Benchmark performance
        classifier.benchmark_performance(test_data)
        
        # Optimize for deployment
        classifier.optimize_for_deployment()
        
        print("Training and optimization complete!")


if __name__ == "__main__":
    main()