from typing import Dict, List, Tuple
import numpy as np
from .analyzers.statistical import StatisticalAnalyzer
from .analyzers.frequency import FrequencyAnalyzer
from .analyzers.pattern import PatternAnalyzer
from .utils.image_loader import ImageLoader
from .utils.result import DetectionResult

class StegoDetector:
    """
    A modular steganography detector that combines multiple analysis techniques
    to detect hidden content in images.
    """
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.image_loader = ImageLoader()
        
    def analyze(self, image_path: str) -> DetectionResult:
        """
        Analyzes an image for potential steganographic content.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DetectionResult object containing analysis results
        """
        # Load and validate image
        image_data = self.image_loader.load(image_path)
        
        # Perform analysis
        statistical_scores = self.statistical_analyzer.analyze(image_data)
        frequency_scores = self.frequency_analyzer.analyze(image_data)
        pattern_scores = self.pattern_analyzer.analyze(image_data)
        
        # Combine results
        result = DetectionResult(
            statistical=statistical_scores,
            frequency=frequency_scores,
            pattern=pattern_scores
        )
        
        return result 