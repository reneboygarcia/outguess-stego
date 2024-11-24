import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging
import warnings
from typing import Dict


class QualityAnalyzer:
    @staticmethod
    def calculate_metrics(
        original: np.ndarray, modified: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various image quality metrics between original and modified images"""
        metrics = {}

        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        metrics["psnr"] = QualityAnalyzer._calculate_psnr(original, modified)

        # Calculate SSIM (Structural Similarity Index)
        metrics["ssim"] = QualityAnalyzer._calculate_ssim(original, modified)

        # Calculate MSE (Mean Squared Error)
        metrics["mse"] = np.mean((original - modified) ** 2)

        # Calculate per-channel MSE metrics
        QualityAnalyzer._calculate_per_channel_metrics(original, modified, metrics)

        # Log the calculated metrics
        QualityAnalyzer._log_metrics(metrics)

        return metrics

    @staticmethod
    def _calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate PSNR while handling potential warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                return psnr(
                    original, modified, data_range=original.max() - original.min()
                )
            except (ZeroDivisionError, RuntimeWarning):
                return float("inf")  # Default to infinity for PSNR
            except Exception as e:
                logging.error(f"Error calculating PSNR: {e}")
                return float("inf")

    @staticmethod
    def _calculate_ssim(original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate SSIM while handling potential errors."""
        try:
            min_dim = min(original.shape[0], original.shape[1])
            win_size = min(
                7, min_dim
            )  # Use the smaller of 7 or the smallest image dimension
            return ssim(original, modified, multichannel=True, win_size=win_size)
        except ValueError as e:
            logging.error(f"Error calculating SSIM: {e}")
            return 1.0 if np.array_equal(original, modified) else 0.0

    @staticmethod
    def _calculate_per_channel_metrics(
        original: np.ndarray, modified: np.ndarray, metrics: Dict[str, float]
    ) -> None:
        """Calculate MSE for each color channel."""
        for i, channel in enumerate(["red", "green", "blue"]):
            channel_mse = np.mean((original[:, :, i] - modified[:, :, i]) ** 2)
            metrics[f"{channel}_mse"] = channel_mse

    @staticmethod
    def _log_metrics(metrics: Dict[str, float]) -> None:
        """Log the calculated metrics."""
        logging.debug("Quality Metrics:")
        for metric, value in metrics.items():
            logging.debug(f"  {metric.upper()}: {value:.4f}")
