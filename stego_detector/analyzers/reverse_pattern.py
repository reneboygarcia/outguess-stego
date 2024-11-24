import numpy as np
from typing import Dict, Tuple, List
import logging
from ..utils.image_data import ImageData

class ReversePatternAnalyzer:
    """
    NSA-grade analyzer specifically targeting known steganographic patterns
    with focus on password-based LSB embedding techniques.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('stego_detector.reverse_pattern')
        
    def analyze(self, image: ImageData) -> Dict[str, float]:
        """
        Advanced pattern analysis targeting password-based embedding
        """
        self.logger.info("Starting NSA-grade reverse pattern analysis")
        results = {}
        
        # Channel analysis to detect preferred embedding channel
        channel_scores = self._analyze_channel_preference(image.raw_data)
        results["channel_signature"] = channel_scores["signature"]
        
        # Password-based position pattern detection
        position_scores = self._detect_position_patterns(image.raw_data)
        results["position_signature"] = position_scores["signature"]
        
        # Length field detection (first 32 bits)
        length_score = self._detect_length_field(image.raw_data)
        results["length_field"] = length_score
        
        # Encryption pattern detection
        crypto_score = self._detect_encryption_patterns(image.raw_data)
        results["encryption_signature"] = crypto_score
        
        return results
    
    def _analyze_channel_preference(self, data: np.ndarray) -> Dict[str, float]:
        """
        Detect channel selection patterns based on known criteria:
        - Variance analysis
        - Edge detection
        - Noise levels
        """
        try:
            if len(data.shape) < 3:
                return {"signature": 0.5}
                
            channel_metrics = []
            for channel in range(data.shape[2]):
                channel_data = data[:,:,channel]
                
                # Calculate metrics used in channel selection
                variance = np.var(channel_data)
                edges = np.sum(np.abs(np.diff(channel_data)))
                noise = self._estimate_noise(channel_data)
                
                # Combine metrics similar to encoder's selection
                channel_score = (variance + edges) / (noise + 1e-10)
                channel_metrics.append(channel_score)
            
            # Look for anomalous channel that matches selection criteria
            channel_ratios = np.array(channel_metrics) / (np.max(channel_metrics) + 1e-10)
            signature = np.std(channel_ratios)
            
            return {
                "signature": min(1.0, signature * 4),
                "likely_channel": np.argmax(channel_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Channel preference analysis failed: {str(e)}")
            return {"signature": 0.5}
    
    def _detect_position_patterns(self, data: np.ndarray) -> Dict[str, float]:
        """
        Detect pseudorandom position patterns from password-based seed
        """
        try:
            # Convert to working channel
            if len(data.shape) == 3:
                channel_scores = self._analyze_channel_preference(data)
                working_channel = data[:,:,channel_scores["likely_channel"]]
            else:
                working_channel = data
            
            # Extract LSB plane
            lsb_plane = working_channel & 1
            
            # Analyze bit positions for pseudorandom patterns
            bit_positions = np.column_stack(np.where(lsb_plane == 1))
            
            if len(bit_positions) == 0:
                return {"signature": 0.0}
            
            # Calculate distances between consecutive positions
            distances = np.diff(bit_positions, axis=0)
            
            # Analyze distance patterns
            distance_hist, _ = np.histogram(distances.flatten(), bins=50)
            distance_entropy = -np.sum((distance_hist/len(distances)) * 
                                     np.log2(distance_hist/len(distances) + 1e-10))
            
            # True random positions would have high entropy
            # Password-based PRNG might show subtle patterns
            normalized_entropy = distance_entropy / np.log2(50)  # Normalize by max entropy
            
            return {
                "signature": 1.0 - normalized_entropy,
                "position_entropy": normalized_entropy
            }
            
        except Exception as e:
            self.logger.error(f"Position pattern detection failed: {str(e)}")
            return {"signature": 0.5}
    
    def _detect_length_field(self, data: np.ndarray) -> float:
        """
        Detect potential message length field in first 32 bits
        """
        try:
            # Get likely embedding channel
            if len(data.shape) == 3:
                channel_scores = self._analyze_channel_preference(data)
                working_channel = data[:,:,channel_scores["likely_channel"]]
            else:
                working_channel = data
            
            # Extract first 32 LSBs (potential length field)
            lsb_plane = working_channel & 1
            first_32_bits = lsb_plane.flatten()[:32]
            
            # Convert to potential length value
            potential_length = 0
            for bit in first_32_bits:
                potential_length = (potential_length << 1) | bit
            
            # Check if length makes sense
            total_pixels = np.prod(data.shape[:2])
            if 0 < potential_length < total_pixels * 0.1:  # Reasonable length check
                return 0.8  # High confidence
            else:
                return 0.2  # Low confidence
                
        except Exception as e:
            self.logger.error(f"Length field detection failed: {str(e)}")
            return 0.5
    
    def _detect_encryption_patterns(self, data: np.ndarray) -> float:
        """
        Detect patterns consistent with encrypted data in LSB plane
        """
        try:
            # Get working channel
            if len(data.shape) == 3:
                channel_scores = self._analyze_channel_preference(data)
                working_channel = data[:,:,channel_scores["likely_channel"]]
            else:
                working_channel = data
            
            # Extract LSB plane
            lsb_plane = working_channel & 1
            
            # Calculate bit sequence metrics
            bit_runs = self._analyze_bit_runs(lsb_plane)
            bit_correlation = self._analyze_bit_correlation(lsb_plane)
            
            # Encrypted data should look random
            # But might have subtle patterns from encryption padding
            randomness_score = (bit_runs + (1 - bit_correlation)) / 2
            
            return min(1.0, max(0.0, randomness_score))
            
        except Exception as e:
            self.logger.error(f"Encryption pattern detection failed: {str(e)}")
            return 0.5
    
    def _estimate_noise(self, channel: np.ndarray) -> float:
        """Estimate noise level in channel"""
        try:
            noise = np.std(channel - np.mean(channel))
            return noise
        except:
            return 0.0
    
    def _analyze_bit_runs(self, bits: np.ndarray) -> float:
        """Analyze runs of consecutive bits"""
        try:
            runs = []
            current_run = 1
            
            flat_bits = bits.flatten()
            for i in range(1, len(flat_bits)):
                if flat_bits[i] == flat_bits[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            
            runs.append(current_run)
            
            # Calculate run length statistics
            run_mean = np.mean(runs)
            run_std = np.std(runs)
            
            # Score based on deviation from expected random runs
            expected_mean = 2.0  # Expected for random bits
            score = 1.0 - np.abs(run_mean - expected_mean) / expected_mean
            
            return score
            
        except Exception:
            return 0.5
    
    def _analyze_bit_correlation(self, bits: np.ndarray) -> float:
        """Analyze bit-to-bit correlation"""
        try:
            flat_bits = bits.flatten()
            correlation = np.corrcoef(flat_bits[:-1], flat_bits[1:])[0,1]
            return np.abs(correlation)
        except:
            return 0.5 