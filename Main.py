#!/usr/bin/env python3
"""
Food Order Verification System
Main application entry point

This program uses computer vision to verify that food orders are being prepared correctly,
by detecting ingredients and comparing them against order specifications.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Import project modules
from food_verification.ui import Application
from food_verification.order_manager import OrderManager
from food_verification.detector import FoodDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"food_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_gpu():
    """Configure TensorFlow to use GPU properly"""
    import tensorflow as tf
    
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Using CPU for inference.")
        return False
    
    # Configure TensorFlow to use all available GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled successfully: {len(gpus)} GPUs available")
        return True
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        return False

# Call this function early
setup_gpu()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Food Order Verification System')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index to use (default: 0)')
    
    parser.add_argument('--model', type=str, default='default',
                        choices=['default', 'efficientdet', 'yolo'],
                        help='Object detection model to use')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to custom model weights (if not using default)')
    
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Confidence threshold for detections (0.0-1.0)')
    
    parser.add_argument('--fullscreen', action='store_true',
                        help='Run application in fullscreen mode')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    
    return parser.parse_args()


def setup_environment():
    """Ensure the application environment is properly set up."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Check for required files
    model_files_exist = True
    
    # Check and download model files if missing
    if not model_files_exist:
        logger.info("Downloading required model files...")
        # Code to download model files would go here
    
    return True


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Food Order Verification System")
    
    # Setup environment and check dependencies
    if not setup_environment():
        logger.error("Failed to setup environment. Exiting.")
        return 1
    
    try:
        # Initialize components
        order_manager = OrderManager('data/orders.json')
        
        detector = FoodDetector(
            model_type=args.model,
            model_path=args.model_path,
            confidence_threshold=args.confidence
        )
        
        # Start the UI application
        app = Application(
            order_manager=order_manager,
            detector=detector,
            camera_index=args.camera,
            fullscreen=args.fullscreen
        )
        
        # Start the application main loop
        app.run()
        
        logger.info("Application exited normally")
        return 0
        
    except Exception as e:
        logger.exception(f"Application failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())