from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """Enhanced NSA-grade detection result analysis"""

    statistical: Dict[str, float]
    frequency: Dict[str, float]
    pattern: Dict[str, float]

    @property
    def channel_analysis(self) -> Dict[str, float]:
        """Analyze potential embedding channels"""
        channel_scores = {}
        
        # Analyze frequency components per channel
        if "fft_noise" in self.frequency:
            channel_scores["high_freq_noise"] = self.frequency["fft_noise"]
            
        # Check pattern distribution
        if "channel_distribution" in self.pattern:
            channel_scores["distribution"] = self.pattern["channel_distribution"]
            
        return channel_scores
    
    @property
    def encryption_signature(self) -> float:
        """Detect encryption patterns"""
        signatures = []
        
        # Check bit randomness
        if "spatial_randomness" in self.pattern:
            signatures.append(self.pattern["spatial_randomness"])
            
        # Check frequency domain artifacts
        if "dct_anomaly" in self.frequency:
            signatures.append(self.frequency["dct_anomaly"])
            
        return np.mean(signatures) if signatures else 0.0
    
    @property
    def embedding_technique(self) -> str:
        """Identify likely embedding technique"""
        if self.encryption_signature > 0.7:
            return "Password-based LSB with encryption"
        elif "lsb_patterns" in self.pattern and self.pattern["lsb_patterns"] > 0.6:
            return "Simple LSB embedding"
        else:
            return "Unknown technique"

    @property
    def overall_probability(self) -> float:
        """Calculate overall probability of hidden content"""
        weights = {"statistical": 0.4, "frequency": 0.35, "pattern": 0.25}

        score = (
            sum(self.statistical.values()) * weights["statistical"]
            + sum(self.frequency.values()) * weights["frequency"]
            + sum(self.pattern.values()) * weights["pattern"]
        ) / (len(self.statistical) + len(self.frequency) + len(self.pattern))

        return min(1.0, max(0.0, score))

    @property
    def verdict(self) -> Tuple[str, str]:
        """Returns (verdict, confidence) based on analysis"""
        prob = self.overall_probability

        if prob > 0.8:
            return "Hidden content highly likely", "High"
        elif prob > 0.6:
            return "Hidden content probable", "Medium"
        elif prob > 0.4:
            return "Possible hidden content", "Low"
        else:
            return "No hidden content detected", "High"
