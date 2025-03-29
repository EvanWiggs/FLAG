"""
COCO Dataset Class Mapping for Food Detection

This file maps standard COCO dataset classes to food ingredients
for use with pre-trained models like SSD MobileNet.
"""

# Standard COCO class mapping (ID -> Name)
COCO_CLASS_MAPPING = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush',
}

# Map COCO classes to food ingredients
# This allows us to translate general object detection into food ingredients
FOOD_INGREDIENT_MAPPING = {
    # Direct food items from COCO
    47: 'banana',       # COCO: banana → ingredient: banana
    48: 'apple',        # COCO: apple → ingredient: apple
    49: 'bread',        # COCO: sandwich → ingredient: bread
    50: 'orange',       # COCO: orange → ingredient: orange
    51: 'broccoli',     # COCO: broccoli → ingredient: broccoli
    52: 'carrot',       # COCO: carrot → ingredient: carrot
    53: 'sausage',      # COCO: hot dog → ingredient: sausage
    54: 'pizza',        # COCO: pizza → ingredient: pizza
    55: 'donut',        # COCO: donut → ingredient: donut
    56: 'cake',         # COCO: cake → ingredient: cake
    
    # Other items that could be used as proxies for ingredients
    20: 'beef',         # COCO: cow → ingredient: beef
    40: 'sauce',        # COCO: bottle → ingredient: sauce
    42: 'drink',        # COCO: cup → ingredient: drink
    43: 'utensil',      # COCO: fork → ingredient: utensil
    44: 'utensil',      # COCO: knife → ingredient: utensil
    45: 'utensil',      # COCO: spoon → ingredient: utensil
    46: 'bowl',         # COCO: bowl → ingredient: bowl
    
    # For testing/demonstration purposes, we'll map some non-food items 
    # to our special ingredients
    15: 'chicken',      # COCO: bird → ingredient: chicken (for demo)
    73: 'pickle',       # COCO: refrigerator → ingredient: pickle (for demo)
    76: 'onion',        # COCO: vase → ingredient: onion (for demo)
    22: 'lettuce',      # COCO: bear → ingredient: lettuce (for demo)
    24: 'tomato',       # COCO: giraffe → ingredient: tomato (for demo)
    1: 'cheese',        # COCO: person → ingredient: cheese (for demo)
}

# These ingredient mappings are just for demonstration purposes.
# In a production system, you would:
# 1. Either train a custom model specifically for your food items
# 2. Or use a food-specific pre-trained model like Food101