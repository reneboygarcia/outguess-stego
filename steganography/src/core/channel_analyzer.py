import numpy as np
from PIL import Image
import logging
from typing import Tuple, Dict
from enum import Enum

class Channel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class ChannelAnalyzer:
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_image(self, image_array: np.ndarray) -> Dict[Channel, float]:
        """
        Analyze each channel of the image and return their suitability scores
        Lower score means better for embedding
        """
        scores = {}
        
        # Split the image into channels
        for channel in Channel:
            channel_data = image_array[:,:,channel.value]
            
            # Calculate metrics for this channel
            variance = np.var(channel_data)
            entropy = self._calculate_entropy(channel_data)
            edge_density = self._calculate_edge_density(channel_data)
            
            # Combine metrics into a single score
            # Lower score means better for embedding
            score = self._calculate_channel_score(variance, entropy, edge_density)
            scores[channel] = score
            
            logging.debug(f"{channel.name} Channel Metrics:")
            logging.debug(f"  Variance: {variance:.2f}")
            logging.debug(f"  Entropy: {entropy:.2f}")
            logging.debug(f"  Edge Density: {edge_density:.2f}")
            logging.debug(f"  Final Score: {score:.2f}")
            
        return scores
    
    def _calculate_entropy(self, channel_data: np.ndarray) -> float:
        """Calculate Shannon entropy of the channel"""
        histogram = np.histogram(channel_data, bins=256, range=(0,256))[0]
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy
    
    def _calculate_edge_density(self, channel_data: np.ndarray) -> float:
        """Calculate the edge density using Sobel operator"""
        dx = np.diff(channel_data, axis=1)
        dy = np.diff(channel_data, axis=0)
        edge_density = (np.abs(dx).mean() + np.abs(dy).mean()) / 2
        return edge_density
    
    def _calculate_channel_score(self, variance: float, entropy: float, edge_density: float) -> float:
        """
        Calculate final score for channel suitability
        Lower score means better for embedding
        """
        # Normalize each metric to 0-1 range
        norm_variance = min(variance / 1000, 1)  # Assuming typical variance range
        norm_entropy = entropy / 8  # Maximum entropy for 8-bit values is 8
        norm_edge_density = min(edge_density / 50, 1)  # Assuming typical edge density range
        
        # Weighted combination of metrics
        # We want:
        # - Higher entropy (more randomness) -> lower score
        # - Lower edge density -> lower score
        # - Moderate variance -> lower score
        score = (
            0.3 * abs(norm_variance - 0.5) +  # Prefer moderate variance
            0.4 * (1 - norm_entropy) +        # Prefer high entropy
            0.3 * norm_edge_density           # Prefer low edge density
        )
        
        return score
    
    def select_best_channel(self, image_path: str) -> Tuple[Channel, np.ndarray]:
        """
        Analyze image and select the best channel for embedding
        Returns: (selected_channel, image_array)
        """
        # Load and convert image
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            image_array = np.array(img, dtype=np.uint8)
        
        # Analyze channels
        scores = self.analyze_image(image_array)
        
        # Select channel with lowest score
        best_channel = min(scores.items(), key=lambda x: x[1])[0]
        logging.info(f"Selected {best_channel.name} channel for embedding (score: {scores[best_channel]:.3f})")
        
        return best_channel, image_array 