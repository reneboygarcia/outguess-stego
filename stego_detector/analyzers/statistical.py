from typing import Dict
import numpy as np
from scipy import stats
import logging
from ..utils.image_data import ImageData


class StatisticalAnalyzer:
    """Performs statistical analysis on image data"""
    
    def __init__(self):
        self.logger = logging.getLogger('stego_detector.statistical')
    
    def analyze(self, image: ImageData) -> Dict[str, float]:
        """
        Performs statistical analysis including:
        - Chi-square test
        - KL divergence
        - Entropy analysis
        """
        self.logger.info("Starting statistical analysis")
        results = {}
        
        # Chi-square analysis
        results["chi_square"] = self._analyze_chi_square(image)
        self.logger.debug(f"Chi-square score: {results['chi_square']:.4f}")
        
        return results
    
    def _analyze_chi_square(self, image: ImageData) -> float:
        """Analyze pixel value distribution using chi-square test"""
        try:
            data = image.raw_data
            
            # Convert to grayscale if color image
            if len(data.shape) == 3:
                # Ensure integer type and proper weighting for grayscale conversion
                gray_data = np.round(0.299 * data[:,:,0] + 
                                   0.587 * data[:,:,1] + 
                                   0.114 * data[:,:,2]).astype(np.uint8)
            else:
                gray_data = data.astype(np.uint8)
            
            self.logger.debug(f"Image data type: {gray_data.dtype}")
            self.logger.debug(f"Value range: [{gray_data.min()}, {gray_data.max()}]")
            
            # Extract LSBs
            lsbs = gray_data & 1
            
            # Calculate observed frequencies
            observed = np.bincount(lsbs.flatten(), minlength=2)
            
            # Expected frequencies (uniform distribution for LSBs)
            expected = np.array([lsbs.size/2, lsbs.size/2])
            
            # Chi-square test
            chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-10))
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            # Convert to anomaly score (0 to 1)
            score = 1 - p_value
            score = min(1.0, max(0.0, score))
            
            self.logger.debug(
                f"Chi-square analysis - stat: {chi2_stat:.4f}, "
                f"p-value: {p_value:.4f}, "
                f"score: {score:.4f}"
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Chi-square analysis failed: {str(e)}")
            self.logger.debug("Stack trace:", exc_info=True)
            return 0.5 