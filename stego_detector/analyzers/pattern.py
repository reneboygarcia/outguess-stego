import numpy as np
from typing import Dict
import logging
from skimage import feature, filters, measure
import cv2
from ..utils.image_data import ImageData

class PatternAnalyzer:
    """Enhanced pattern analyzer using advanced image processing techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger('stego_detector.pattern')
    
    def analyze(self, image: ImageData) -> Dict[str, float]:
        """
        Enhanced pattern analysis including:
        - LSB pattern analysis
        - Texture analysis
        - Edge coherence
        - Local binary patterns
        """
        self.logger.info("Starting enhanced pattern analysis")
        results = {}
        
        # Basic LSB analysis
        results["lsb_patterns"] = self._analyze_lsb_patterns(image.raw_data)
        
        # Texture analysis
        results["texture"] = self._analyze_texture(image.raw_data)
        
        # Edge coherence
        results["edge_coherence"] = self._analyze_edge_coherence(image.raw_data)
        
        # Local binary patterns
        results["lbp_uniformity"] = self._analyze_lbp(image.raw_data)
        
        # Channel distribution analysis
        results["channel_distribution"] = self._analyze_channel_distribution(image.raw_data)
        
        # Spatial randomness analysis
        results["spatial_randomness"] = self._analyze_spatial_randomness(image.raw_data)
        
        self.logger.info(f"Pattern analysis results: {results}")
        return results
    
    def _analyze_lsb_patterns(self, data: np.ndarray) -> float:
        """Enhanced LSB pattern analysis"""
        try:
            if len(data.shape) == 3:
                gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray_data = data
            
            # Extract LSBs
            lsbs = gray_data & 1
            
            # Calculate bit transitions with sliding window
            window_size = 3
            scores = []
            
            for i in range(0, lsbs.shape[0] - window_size + 1):
                for j in range(0, lsbs.shape[1] - window_size + 1):
                    window = lsbs[i:i+window_size, j:j+window_size]
                    
                    # Calculate local transition density
                    h_trans = np.sum(np.abs(np.diff(window, axis=1)))
                    v_trans = np.sum(np.abs(np.diff(window, axis=0)))
                    
                    density = (h_trans + v_trans) / (2 * window.size)
                    scores.append(abs(0.5 - density))
            
            score = np.mean(scores)
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"LSB pattern analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_texture(self, data: np.ndarray) -> float:
        """Analyze texture patterns using GLCM"""
        try:
            if len(data.shape) == 3:
                gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray_data = data
            
            # Calculate GLCM properties
            glcm = feature.graycomatrix(
                gray_data, 
                distances=[1], 
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                symmetric=True,
                normed=True
            )
            
            # Calculate texture properties
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            dissim = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            corr = feature.graycoprops(glcm, 'correlation')[0, 0]
            
            # Combine metrics
            score = (contrast + dissim + (1 - corr)) / 3
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Texture analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_edge_coherence(self, data: np.ndarray) -> float:
        """Analyze edge coherence and continuity"""
        try:
            if len(data.shape) == 3:
                gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray_data = data
            
            # Calculate edges using multiple methods
            edges_sobel = filters.sobel(gray_data)
            edges_canny = feature.canny(gray_data)
            
            # Analyze edge consistency
            edge_diff = np.abs(edges_sobel - edges_canny)
            coherence = 1.0 - np.mean(edge_diff)
            
            return min(1.0, max(0.0, coherence))
            
        except Exception as e:
            self.logger.error(f"Edge coherence analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_lbp(self, data: np.ndarray) -> float:
        """Analyze local binary patterns"""
        try:
            if len(data.shape) == 3:
                gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray_data = data
            
            # Calculate LBP
            radius = 1
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(
                gray_data, 
                n_points, 
                radius, 
                method='uniform'
            )
            
            # Calculate LBP histogram
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
            
            # Calculate uniformity
            uniformity = np.sum(hist ** 2)
            score = 1.0 - uniformity  # Less uniform = higher probability of hidden content
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"LBP analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_channel_distribution(self, data: np.ndarray) -> float:
        """Analyze distribution of modifications across channels"""
        try:
            if len(data.shape) < 3:
                return 0.5
            
            channel_scores = []
            for channel in range(data.shape[2]):
                # Extract LSBs for each channel
                lsbs = data[:,:,channel] & 1
                
                # Calculate local density variations
                window_size = 16  # Larger window to detect patterns
                density_map = np.zeros((data.shape[0]//window_size, data.shape[1]//window_size))
                
                for i in range(density_map.shape[0]):
                    for j in range(density_map.shape[1]):
                        window = lsbs[i*window_size:(i+1)*window_size, 
                                    j*window_size:(j+1)*window_size]
                        density_map[i,j] = np.mean(window)
                
                # Calculate variation in densities
                density_std = np.std(density_map)
                density_entropy = -np.sum(density_map * np.log2(density_map + 1e-10))
                
                # Combine metrics
                channel_score = (density_std + density_entropy/8) / 2
                channel_scores.append(channel_score)
            
            # Look for anomalous channel
            channel_variations = np.std(channel_scores)
            return min(1.0, channel_variations * 4)  # Scale up subtle variations
            
        except Exception as e:
            self.logger.error(f"Channel distribution analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_spatial_randomness(self, data: np.ndarray) -> float:
        """Detect pseudorandom patterns in LSB plane"""
        try:
            if len(data.shape) == 3:
                gray_data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            else:
                gray_data = data
            
            lsbs = gray_data & 1
            
            # Calculate autocorrelation at multiple offsets
            max_offset = 32
            correlations = []
            
            for offset in range(1, max_offset):
                # Horizontal correlation
                h_corr = np.corrcoef(lsbs[:,:-offset].flatten(), 
                                    lsbs[:,offset:].flatten())[0,1]
                # Vertical correlation
                v_corr = np.corrcoef(lsbs[:-offset,:].flatten(), 
                                    lsbs[offset:,:].flatten())[0,1]
                correlations.extend([h_corr, v_corr])
            
            # True random data should have very low correlations
            # Pseudorandom might show subtle patterns
            correlations = np.abs(correlations)
            score = np.mean(correlations) * 2  # Scale up subtle correlations
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Spatial randomness analysis failed: {str(e)}")
            return 0.5