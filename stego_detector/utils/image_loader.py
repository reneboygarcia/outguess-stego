from PIL import Image
import numpy as np
from .image_data import ImageData

class ImageLoader:
    """Handles image loading and validation"""
    
    def load(self, path: str) -> ImageData:
        """Load and validate an image file"""
        try:
            return ImageData.from_path(path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}") 