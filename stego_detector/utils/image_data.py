from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class ImageData:
    """Container for image data and metadata"""
    raw_data: np.ndarray
    color_space: str
    format: str
    
    @classmethod
    def from_path(cls, path: str) -> 'ImageData':
        """Create ImageData from an image file"""
        img = Image.open(path)
        return cls(
            raw_data=np.array(img),
            color_space='RGB',
            format=img.format
        ) 