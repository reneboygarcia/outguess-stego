"""
Steganography package for hiding and extracting messages in images.
"""

from steganography.src.core import (
    ImageEncoder,
    ImageDecoder,
    ChannelAnalyzer,
    Channel,
    ErrorCorrection,
    QualityAnalyzer,
    OutguessStego
)

__version__ = '0.1'
__all__ = [
    'ImageEncoder',
    'ImageDecoder',
    'ChannelAnalyzer',
    'Channel',
    'ErrorCorrection',
    'QualityAnalyzer',
    'OutguessStego'
]
