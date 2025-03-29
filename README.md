# Food Order Verification System

This system uses computer vision to verify food orders during preparation by detecting ingredients via webcam and comparing them against order specifications to flag any discrepancies.

## Features

- Real-time food ingredient detection using computer vision
- Customizable order specifications (required and forbidden ingredients)
- Visual flagging of order violations (e.g., detecting pickles on a "no pickles" sandwich)
- Simple user interface for monitoring food preparation
- Order management system

## Requirements

- Python 3.8 or higher
- Webcam or camera connected to the computer
- Python packages listed in `requirements.txt`

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python main.py
   ```

## Usage

1. **Starting the application**:
   - Run `python main.py` to launch the application
   - The application will automatically load sample orders

2. **Camera controls**:
   - Click "Start Camera" to begin the video feed
   - Use the Camera menu to select a different camera if needed
   - Click "Capture Frame" to save the current frame

3. **Order navigation**:
   - Use "Previous Item" and "Next Item" buttons to navigate through items in the current order
   - The system will automatically check for violations based on the selected item

4. **Verification process**:
   - The system will detect ingredients in the video feed in real-time
   - If a forbidden ingredient is detected, it will be highlighted in red
   - Violations will be displayed in the Violations panel

## Command Line Options

The application supports several command line options:

```
usage: main.py [-h] [--camera CAMERA] [--model {default,efficientdet,yolo}]
               [--model-path MODEL_PATH] [--confidence CONFIDENCE]
               [--fullscreen] [--debug]

Food Order Verification System

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA       Camera index to use (default: 0)
  --model {default,efficientdet,yolo}
                        Object detection model to use
  --model-path MODEL_PATH
                        Path to custom model weights (if not using default)
  --confidence CONFIDENCE
                        Confidence threshold for detections (0.0-1.0)
  --fullscreen          Run application in fullscreen mode
  --debug               Enable debug mode with additional logging
```

## Project Structure

- `main.py`: Main application entry point
- `food_verification/`: Core package
  - `detector.py`: Computer vision and detection functionality
  - `order_manager.py`: Order management and verification
  - `ui.py`: User interface components
- `models/`: Directory for model files (automatically downloaded)
- `data/`: Directory for order data
- `captures/`: Directory for captured frames

## Customization

### Adding Custom Food Classes

To add additional food items to the detection system, modify the `FOOD_CLASS_MAPPING` dictionary in `food_verification/detector.py`:

```python
FOOD_CLASS_MAPPING = {
    1: 'bread',
    2: 'chicken',
    # Add more mappings here
    16: 'avocado',
    17: 'mushroom',
}
```

### Using a Custom Model

You can use a custom-trained detection model by specifying it at startup:

```
python main.py --model-path /path/to/your/model --model custom
```

## Troubleshooting

### Camera Issues

- If the camera doesn't start, try a different camera index: `python main.py --camera 1`
- Make sure no other application is using the camera

### Detection Problems

- If detection is inaccurate, you may need a custom-trained model for your specific food items
- Try adjusting the confidence threshold: `python main.py --confidence 0.7`

## License

This software is provided as-is for educational and demonstration purposes.

## Future Improvements

- Integration with POS systems
- Support for multiple camera views
- Training custom models for specific menu items
- Integration with kitchen display systems
- Analytics for common preparation errors