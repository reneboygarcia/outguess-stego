import numpy as np
from typing import Dict, List, Optional
import logging
from ..utils.image_data import ImageData

class EncoderCracker:
    """NSA-grade encoder cracking tool"""
    
    def __init__(self):
        self.logger = logging.getLogger('stego_detector.cracker')
        
    def crack(self, image: ImageData) -> Dict:
        """
        Attempt to crack the encoder's implementation
        """
        results = {}
        
        # 1. Identify embedding channel
        channel_info = self._identify_channel(image.raw_data)
        results["channel"] = channel_info
        
        # 2. Extract potential length field
        length_info = self._extract_length_field(
            image.raw_data[:,:,channel_info["likely_channel"]]
        )
        results["length"] = length_info
        
        # 3. Analyze embedding pattern
        pattern_info = self._analyze_embedding_pattern(
            image.raw_data[:,:,channel_info["likely_channel"]],
            length_info["likely_length"]
        )
        results["pattern"] = pattern_info
        
        # 4. Detect encryption method
        crypto_info = self._detect_encryption_method(
            image.raw_data[:,:,channel_info["likely_channel"]]
        )
        results["encryption"] = crypto_info
        
        return results
    
    def _identify_channel(self, data: np.ndarray) -> Dict:
        """Identify the likely embedding channel"""
        if len(data.shape) < 3:
            return {"likely_channel": 0}
            
        channel_scores = []
        for i in range(data.shape[2]):
            channel = data[:,:,i]
            
            # Calculate metrics the encoder likely uses
            variance = np.var(channel)
            edges = np.sum(np.abs(np.diff(channel)))
            noise = np.std(channel - np.mean(channel))
            
            # Combine metrics similar to encoder
            score = (variance + edges) / (noise + 1e-10)
            channel_scores.append(score)
        
        likely_channel = np.argmax(channel_scores)
        confidence = channel_scores[likely_channel] / sum(channel_scores)
        
        return {
            "likely_channel": likely_channel,
            "confidence": confidence,
            "channel_scores": channel_scores
        }
    
    def _extract_length_field(self, channel: np.ndarray) -> Dict:
        """Extract potential message length field"""
        # Try different length field positions
        candidates = []
        
        # Linear first 32 bits
        linear_bits = channel.flatten()[:32] & 1
        linear_length = self._bits_to_int(linear_bits)
        
        # First row
        row_bits = channel[0,:32] & 1
        row_length = self._bits_to_int(row_bits)
        
        # First column
        col_bits = channel[:32,0] & 1
        col_length = self._bits_to_int(col_bits)
        
        # Check which length makes most sense
        total_pixels = channel.size
        for length, source in [(linear_length, "linear"), 
                             (row_length, "row"), 
                             (col_length, "column")]:
            if 0 < length < total_pixels * 0.5:  # Reasonable length check
                candidates.append((length, source))
        
        if candidates:
            likely_length, source = min(candidates, key=lambda x: x[0])
            return {
                "likely_length": likely_length,
                "source": source,
                "candidates": candidates
            }
        return {"likely_length": None}
    
    def _analyze_embedding_pattern(self, channel: np.ndarray, length: Optional[int]) -> Dict:
        """Analyze the embedding pattern used"""
        lsbs = channel & 1
        
        # Look for systematic patterns
        row_pattern = np.mean(lsbs, axis=1)
        col_pattern = np.mean(lsbs, axis=0)
        
        # Check for password-based positioning
        positions = np.column_stack(np.where(lsbs == 1))
        
        if len(positions) > 1:
            # Analyze distances between modified bits
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            distance_entropy = self._calculate_entropy(distances)
            
            # High entropy suggests password-based positioning
            if distance_entropy > 0.8:
                return {
                    "type": "password_based",
                    "entropy": distance_entropy
                }
        
        return {
            "type": "sequential",
            "row_pattern": row_pattern,
            "col_pattern": col_pattern
        }
    
    def _detect_encryption_method(self, channel: np.ndarray) -> Dict:
        """Detect encryption method used"""
        lsbs = channel & 1
        
        # Analyze bit patterns
        bit_runs = self._analyze_bit_runs(lsbs)
        correlations = self._analyze_correlations(lsbs)
        
        # Check for Fernet encryption signatures
        if self._check_fernet_signature(lsbs):
            return {
                "method": "fernet",
                "confidence": "high"
            }
        
        # Check for other common encryption methods
        if bit_runs["randomness"] > 0.8 and correlations["score"] < 0.2:
            return {
                "method": "strong_encryption",
                "confidence": "medium"
            }
            
        return {
            "method": "unknown",
            "confidence": "low"
        }
    
    def _bits_to_int(self, bits: np.ndarray) -> int:
        """Convert bit array to integer"""
        return int(''.join(map(str, bits)), 2)
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        hist = np.histogram(data, bins='auto', density=True)[0]
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _analyze_bit_runs(self, bits: np.ndarray) -> Dict:
        """Analyze runs of consecutive bits"""
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
        
        return {
            "mean_length": np.mean(runs),
            "std_length": np.std(runs),
            "randomness": 1.0 - abs(np.mean(runs) - 2.0) / 2.0
        }
    
    def _analyze_correlations(self, bits: np.ndarray) -> Dict:
        """Analyze bit correlations"""
        flat_bits = bits.flatten()
        correlations = []
        
        for offset in range(1, min(32, len(flat_bits))):
            corr = np.corrcoef(flat_bits[:-offset], flat_bits[offset:])[0,1]
            correlations.append(abs(corr))
            
        return {
            "score": np.mean(correlations),
            "max_correlation": max(correlations)
        }
    
    def _check_fernet_signature(self, bits: np.ndarray) -> bool:
        """Check for Fernet encryption signatures"""
        # Fernet typically has a specific header pattern
        flat_bits = bits.flatten()
        if len(flat_bits) < 64:
            return False
            
        # Check for potential Fernet version and timestamp patterns
        header_bits = flat_bits[:64]
        version_bits = header_bits[:8]
        timestamp_bits = header_bits[8:64]
        
        # Fernet version 128 (0x80) check
        if self._bits_to_int(version_bits) == 128:
            return True
            
        return False 