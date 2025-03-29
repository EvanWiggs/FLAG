
"""
Food Detector Module

This module handles all computer vision and detection functionality
using TensorFlow and OpenCV to identify food items in the video stream.
"""

import os
import logging
import time
import threading
import queue
import urllib.request
import zipfile
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np
import tensorflow as tf

# Import COCO mappings
from food_verification.coco_mapping import COCO_CLASS_MAPPING, FOOD_INGREDIENT_MAPPING

logger = logging.getLogger(__name__)

# Default paths for models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'efficientdet_lite0_food')


class DetectionResult:
    """Class representing a detection result."""
    
    def __init__(self, 
                 class_id: int, 
                 class_name: str, 
                 confidence: float, 
                 box: Tuple[int, int, int, int]):
        """
        Initialize a detection result.
        
        Args:
            class_id: Numeric ID of the detected class
            class_name: Name of the detected class/ingredient
            confidence: Confidence score (0.0-1.0)
            box: Bounding box coordinates (x, y, width, height)
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.box = box  # (x, y, width, height)
    
    def __repr__(self):
        return f"DetectionResult({self.class_name}, {self.confidence:.2f}, {self.box})"


class FoodDetector:
    """
    Food Detector class responsible for detecting food items using computer vision.
    """
    
    def __init__(self,
                 model_type: str = 'default',
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.6,
                 max_queue_size: int = 5):
        """
        Initialize the food detector.
        
        Args:
            model_type: Type of model to use ('default', 'efficientdet', 'yolo')
            model_path: Path to custom model weights (if not using default)
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            max_queue_size: Maximum size of the processing queue
        """
        self.model_type = model_type
        self.model_path = model_path if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        
        # Create directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Initialize state variables
        self.is_running = False
        self.model = None
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_thread = None
        
        # Load the detection model
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate object detection model."""
        if not os.path.exists(self.model_path):
            logger.info(f"Model not found at {self.model_path}, attempting to download...")
            self._download_model()
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model with TensorFlow 2.19.0 approach
            # This should work with the SSD MobileNet model
            self.model = tf.saved_model.load(self.model_path)
            
            # Set up the detection function
            self.detect_fn = self.model.signatures['serving_default']
            
            # Test with a dummy image to ensure it's working
            logger.info("Testing model with dummy image...")
            dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
            self._run_inference_for_single_image(dummy_image)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _download_model(self):
        """Download the model if it doesn't exist."""
        logger.warning("Automatic model download not implemented")
        logger.info("Please run the direct_download.py script to download the model")
        raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def _run_inference_for_single_image(self, image):
        """
        Run inference for a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            
        Returns:
            Dictionary of detection results
        """
        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image)
        
        # Add batch dimension
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run inference
        detection_output = self.detect_fn(input_tensor)
        
        # Create a dictionary with the expected format
        output_dict = {}
        
        # Convert outputs to numpy arrays
        for key, value in detection_output.items():
            output_dict[key] = value.numpy()
        
        # For SSD models, detection_classes should be a 2D array, but we need a 1D array
        if 'detection_classes' in output_dict and len(output_dict['detection_classes'].shape) > 1:
            output_dict['detection_classes'] = output_dict['detection_classes'][0]
            
        # Same for detection_scores
        if 'detection_scores' in output_dict and len(output_dict['detection_scores'].shape) > 1:
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            
        # And for detection_boxes
        if 'detection_boxes' in output_dict and len(output_dict['detection_boxes'].shape) > 1:
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        
        # Add num_detections if it's not present
        if 'num_detections' not in output_dict:
            if 'detection_classes' in output_dict:
                output_dict['num_detections'] = len(output_dict['detection_classes'])
            else:
                output_dict['num_detections'] = 0
                
        # SSD MobileNet models typically have a fixed number of detections, even if most are low confidence
        # So we'll filter by confidence later
                
        # Convert detection classes to integers
        if 'detection_classes' in output_dict:
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        
        return output_dict
    
    def _process_frame(self, frame):
        """
        Process a single frame to detect food items.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            List of DetectionResult objects
        """
        # Resize image if needed
        image_np = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB (TensorFlow models expect RGB)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Run inference
        output_dict = self._run_inference_for_single_image(image_np)
        
        # Process results
        results = []
        
        # Check if we have detection results with the expected keys
        if 'detection_classes' not in output_dict or 'detection_scores' not in output_dict:
            logger.warning("Missing expected keys in detection results")
            return results
            
        # Process each detection
        for i in range(int(output_dict['num_detections'])):
            try:
                # Get confidence score
                confidence = float(output_dict['detection_scores'][i])
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class ID
                # SSD MobileNet uses 1-indexed classes directly
                class_id = int(output_dict['detection_classes'][i])
                
                # Determine ingredient based on class ID
                if class_id in FOOD_INGREDIENT_MAPPING:
                    # This is a food ingredient we care about
                    ingredient_name = FOOD_INGREDIENT_MAPPING[class_id]
                elif class_id in COCO_CLASS_MAPPING:
                    # This is a recognized object, but not a food ingredient
                    # Skip for order verification
                    continue
                else:
                    # Unknown class ID, skip
                    continue
                
                # Get bounding box
                if 'detection_boxes' in output_dict:
                    box = output_dict['detection_boxes'][i]
                    
                    # Convert normalized coordinates to pixel coordinates
                    height, width, _ = image_np.shape
                    y1, x1, y2, x2 = box
                    
                    x = int(x1 * width)
                    y = int(y1 * height)
                    w = int((x2 - x1) * width)
                    h = int((y2 - y1) * height)
                    
                    # Create and add detection result
                    result = DetectionResult(
                        class_id=class_id,
                        class_name=ingredient_name,
                        confidence=confidence,
                        box=(x, y, w, h)
                    )
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing detection {i}: {e}")
        
        return results
    
    def _process_frames_thread(self):
        """Background thread for processing frames."""
        logger.info("Processing thread started")
        
        while self.is_running:
            try:
                # Get a frame from the queue with a timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process the frame
                results = self._process_frame(frame)
                
                # Put results in the output queue
                self.result_queue.put((frame, results))
                
                # Mark the task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
        
        logger.info("Processing thread stopped")
    
    def start(self):
        """Start the detector processing thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._process_frames_thread,
            daemon=True
        )
        self.processing_thread.start()
    
    def stop(self):
        """Stop the detector processing thread."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
    
    def process_frame(self, frame):
        """
        Add a frame to the processing queue.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            True if frame was added to queue, False otherwise
        """
        if not self.is_running:
            return False
        
        try:
            # Try to add the frame to the queue without blocking
            self.frame_queue.put_nowait(frame.copy())
            return True
        except queue.Full:
            # Queue is full, skip this frame
            return False
    
    def get_results(self):
        """
        Get the latest detection results.
        
        Returns:
            Tuple of (frame, list of DetectionResult objects) or None if no results available
        """
        try:
            # Try to get results without blocking
            return self.result_queue.get_nowait()
        except queue.Empty:
            # No results available
            return None
    
    def check_violations(self, results, order_item):
        """
        Check detection results against order specifications for violations.
        
        Args:
            results: List of DetectionResult objects
            order_item: OrderItem object containing requirements
            
        Returns:
            Tuple of (set of violations, bool indicating if verification is complete)
        """
        if not results:
            return set(), False
        
        # Extract detected ingredients
        detected_ingredients = {result.class_name for result in results}
        
        # Check for violations (forbidden ingredients)
        violations = set()
        for forbidden in order_item.forbidden_ingredients:
            if forbidden in detected_ingredients:
                violations.add(forbidden)
        
        # Check if all required ingredients are present
        missing_required = set()
        for required in order_item.required_ingredients:
            if required not in detected_ingredients:
                missing_required.add(required)
        
        # Consider verification complete when all required ingredients are present
        # In a real system, you would have a more sophisticated completion check
        verification_complete = (len(missing_required) == 0)
        
        return violations, verification_complete

    def annotate_frame(self, frame, results, violations=None):
        """
        Annotate a frame with detection results and violations.
        
        Args:
            frame: Input frame
            results: List of DetectionResult objects
            violations: Set of ingredients that violate the order requirements
            
        Returns:
            Annotated frame
        """
        if violations is None:
            violations = set()
        
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw detection boxes and labels
        for result in results:
            x, y, w, h = result.box
            
            # Determine color (red for violations, green for valid ingredients)
            if result.class_name in violations:
                color = (0, 0, 255)  # Red (BGR)
            else:
                color = (0, 255, 0)  # Green (BGR)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            text = f"{result.class_name} ({result.confidence:.2f})"
            
            # Draw background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                annotated_frame,
                (x, y - text_size[1] - 10),
                (x + text_size[0], y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                2
            )
        
        # Draw violation warning if any violations exist
        if violations:
            warning_text = "VIOLATIONS DETECTED!"
            cv2.putText(
                annotated_frame,
                warning_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # Red
                2
            )
            
            # List the violations
            y_offset = 60
            for violation in violations:
                cv2.putText(
                    annotated_frame,
                    f"- {violation} should not be present",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red
                    2
                )
                y_offset += 25
        
        return annotated_frame
