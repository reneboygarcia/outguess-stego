import numpy as np
from scipy import fftpack
from typing import Dict
from ..utils.image_data import ImageData

class FrequencyAnalyzer:
    """
    Analyzes frequency domain characteristics of images to detect steganography
    using DCT (Discrete Cosine Transform) and FFT (Fast Fourier Transform).
    """
    
    def analyze(self, image: ImageData) -> Dict[str, float]:
        """
        Performs frequency domain analysis including:
        - DCT coefficient analysis
        - FFT magnitude spectrum analysis
        - High-frequency noise analysis
        """
        results = {}
        
        # Convert to grayscale if RGB
        if len(image.raw_data.shape) == 3:
            gray_data = np.mean(image.raw_data, axis=2)
        else:
            gray_data = image.raw_data
            
        # DCT analysis
        results["dct_anomaly"] = self._analyze_dct(gray_data)
        
        # FFT analysis
        results["fft_noise"] = self._analyze_fft(gray_data)
        
        # Block artifact analysis
        results["block_artifacts"] = self._analyze_block_artifacts(gray_data)
        
        return results
    
    def _analyze_dct(self, data: np.ndarray) -> float:
        """Analyze DCT coefficients for unusual patterns"""
        try:
            # Apply DCT
            dct = fftpack.dct(fftpack.dct(data.T, norm='ortho').T, norm='ortho')
            
            # Analyze coefficient distribution
            dct_flat = dct.flatten()
            
            # Calculate ratio of high to low frequency components
            threshold = np.percentile(np.abs(dct_flat), 70)
            high_freq = np.sum(np.abs(dct_flat) > threshold)
            total = len(dct_flat)
            
            ratio = high_freq / total
            # Normalize to [0,1]
            score = min(1.0, ratio * 2.5)
            
            return score
            
        except Exception:
            return 0.5
    
    def _analyze_fft(self, data: np.ndarray) -> float:
        """Analyze FFT spectrum for unusual noise patterns"""
        try:
            # Apply FFT
            fft = np.fft.fft2(data)
            magnitude = np.abs(np.fft.fftshift(fft))
            
            # Calculate high-frequency energy ratio
            total_energy = np.sum(magnitude)
            center_mask = self._create_center_mask(magnitude.shape)
            high_freq_energy = np.sum(magnitude * (1 - center_mask))
            
            ratio = high_freq_energy / total_energy
            # Normalize to [0,1]
            score = min(1.0, ratio * 3.0)
            
            return score
            
        except Exception:
            return 0.5
    
    def _analyze_block_artifacts(self, data: np.ndarray) -> float:
        """Detect block artifacts that might indicate steganography"""
        try:
            # Calculate differences between adjacent pixels
            diff_h = np.abs(data[:, 1:] - data[:, :-1])
            diff_v = np.abs(data[1:, :] - data[:-1, :])
            
            # Look for periodic patterns
            h_periodic = self._detect_periodicity(np.mean(diff_h, axis=0))
            v_periodic = self._detect_periodicity(np.mean(diff_v, axis=1))
            
            score = (h_periodic + v_periodic) / 2
            return score
            
        except Exception:
            return 0.5
    
    def _create_center_mask(self, shape: tuple, radius_factor: float = 0.1) -> np.ndarray:
        """Create a circular mask for the center (low) frequencies"""
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        radius = int(min(rows, cols) * radius_factor)
        
        y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
        mask = x*x + y*y <= radius*radius
        
        return mask.astype(float)
    
    def _detect_periodicity(self, signal: np.ndarray) -> float:
        """Detect periodic patterns in a signal"""
        try:
            # Calculate autocorrelation
            correlation = np.correlate(signal, signal, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Find peaks
            peaks = self._find_peaks(correlation)
            
            if len(peaks) < 2:
                return 0.0
                
            # Calculate regularity of peak spacing
            peak_spaces = np.diff(peaks)
            regularity = 1.0 - np.std(peak_spaces) / np.mean(peak_spaces)
            
            return max(0.0, min(1.0, regularity))
            
        except Exception:
            return 0.0
    
    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Find peaks in a signal above a threshold"""
        peak_indices = []
        for i in range(1, len(signal)-1):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                signal[i] > threshold * np.max(signal)):
                peak_indices.append(i)
        return np.array(peak_indices) 