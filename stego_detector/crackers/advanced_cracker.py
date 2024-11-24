import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from ..utils.image_data import ImageData
from sklearn.cluster import DBSCAN


class AdvancedCracker:
    """Advanced Food-grade steganography cracking tool"""

    def __init__(self):
        self.logger = logging.getLogger("stego_detector.advanced_cracker")

    def analyze(self, image: ImageData) -> Dict:
        """Perform advanced steganalysis"""
        results = {}

        # Deep channel analysis
        channel_info = self._analyze_channels(image.raw_data)
        results["channels"] = channel_info

        # If we found a likely channel, analyze it deeply
        if channel_info["most_likely"] is not None:
            channel = image.raw_data[:, :, channel_info["most_likely"]]

            # Analyze LSB patterns
            lsb_info = self._analyze_lsb_patterns(channel)
            results["lsb_analysis"] = lsb_info

            # Look for length field
            length_info = self._find_length_field(
                channel, lsb_info["modified_positions"]
            )
            results["length_field"] = length_info

            # Analyze encryption signatures
            crypto_info = self._analyze_encryption(
                channel, length_info["likely_offset"]
            )
            results["encryption"] = crypto_info

            # Password-based pattern analysis
            pattern_info = self._analyze_password_patterns(
                channel, lsb_info["modified_positions"]
            )
            results["password_patterns"] = pattern_info

        return results

    def _analyze_channels(self, data: np.ndarray) -> Dict:
        """Advanced channel analysis"""
        if len(data.shape) < 3:
            return {"most_likely": None}

        channel_metrics = []
        for i in range(data.shape[2]):
            channel = data[:, :, i]
            metrics = {
                "variance": np.var(channel),
                "entropy": self._calculate_entropy(channel & 1),
                "chi_square": self._chi_square_test(channel & 1),
                "runs_test": self._runs_test(channel & 1),
                "autocorr": self._autocorrelation(channel & 1),
            }
            channel_metrics.append(metrics)

        # Score channels based on steganographic likelihood
        scores = []
        for metrics in channel_metrics:
            score = (
                metrics["entropy"] * 0.3
                + metrics["chi_square"] * 0.3
                + metrics["runs_test"] * 0.2
                + (1 - abs(metrics["autocorr"])) * 0.2
            )
            scores.append(score)

        most_likely = np.argmax(scores)
        confidence = scores[most_likely] / sum(scores)

        return {
            "most_likely": most_likely,
            "confidence": confidence,
            "channel_scores": scores,
            "metrics": channel_metrics,
        }

    def _analyze_lsb_patterns(self, channel: np.ndarray) -> Dict:
        """Detailed LSB pattern analysis"""
        lsbs = channel & 1
        modified_positions = np.where(lsbs == 1)

        # Analyze local density variations
        density_map = self._calculate_density_map(lsbs)

        # Look for structured patterns
        pattern_score = self._detect_structured_patterns(modified_positions)

        # Calculate distribution metrics
        distribution = self._analyze_bit_distribution(modified_positions)

        return {
            "modified_positions": modified_positions,
            "density_variations": np.std(density_map),
            "pattern_score": pattern_score,
            "distribution": distribution,
        }

    def _find_length_field(
        self, channel: np.ndarray, modified_positions: Tuple[np.ndarray, np.ndarray]
    ) -> Dict:
        """Advanced length field detection"""
        try:
            candidates = []

            # Try different starting positions
            start_positions = [
                (0, 0),  # Top-left corner
                (0, channel.shape[1] // 2),  # Top-middle
                (channel.shape[0] // 2, 0),  # Middle-left
                # Safely handle first modified position
                (modified_positions[0][0] if len(modified_positions[0]) > 0 else 0, 0),
            ]

            for start_x, start_y in start_positions:
                # Extract potential length field (32 bits)
                length_bits = np.zeros(32, dtype=np.int8)
                for i in range(32):
                    x = (start_x + i) % channel.shape[0]
                    y = (start_y + i) % channel.shape[1]
                    length_bits[i] = channel[x, y] & 1

                length = self._bits_to_int(length_bits)
                if self._is_valid_length(length, channel.size):
                    candidates.append(
                        {
                            "length": length,
                            "offset": (start_x, start_y),
                            "confidence": self._assess_length_confidence(
                                length, channel
                            ),
                        }
                    )

            # Select most likely candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: x["confidence"])
                return {
                    "likely_length": best_candidate["length"],
                    "likely_offset": best_candidate["offset"],
                    "confidence": best_candidate["confidence"],
                    "candidates": candidates,
                }

            return {
                "likely_length": None,
                "likely_offset": None,
                "confidence": 0.0,
                "candidates": [],
            }

        except Exception as e:
            self.logger.error(f"Length field detection failed: {str(e)}")
            return {
                "likely_length": None,
                "likely_offset": None,
                "confidence": 0.0,
                "candidates": [],
            }

    def _analyze_encryption(
        self, channel: np.ndarray, length_offset: Optional[Tuple[int, int]]
    ) -> Dict:
        """Advanced encryption analysis"""
        if length_offset is None:
            lsbs = channel & 1
        else:
            # Analyze bits after length field
            x, y = length_offset
            lsbs = (channel[x:, y:] & 1).flatten()

        # Statistical tests for randomness
        randomness_tests = {
            "chi_square": self._chi_square_test(lsbs),
            "runs": self._runs_test(lsbs),
            "autocorr": self._autocorrelation(lsbs),
            "entropy": self._calculate_entropy(lsbs),
        }

        # Look for known encryption signatures
        signatures = self._detect_encryption_signatures(lsbs)

        # Combine evidence
        encryption_score = (
            randomness_tests["chi_square"] * 0.3
            + randomness_tests["runs"] * 0.3
            + (1 - abs(randomness_tests["autocorr"])) * 0.2
            + (randomness_tests["entropy"] / 8.0) * 0.2
        )

        return {
            "likely_encrypted": encryption_score > 0.7,
            "encryption_score": encryption_score,
            "randomness_tests": randomness_tests,
            "signatures": signatures,
        }

    def _analyze_password_patterns(
        self, channel: np.ndarray, modified_positions: Tuple[np.ndarray, np.ndarray]
    ) -> Dict:
        """Analyze patterns that might indicate password-based positioning"""
        if len(modified_positions[0]) < 2:
            return {"password_based": False}

        # Calculate distances between consecutive positions
        positions = np.column_stack(modified_positions)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))

        # Analyze distance distribution
        distance_entropy = self._calculate_entropy(distances)
        distance_stats = {
            "mean": np.mean(distances),
            "std": np.std(distances),
            "min": np.min(distances),
            "max": np.max(distances),
        }

        # Look for PRNG patterns
        prng_score = self._detect_prng_patterns(distances)

        return {
            "password_based": prng_score > 0.7,
            "prng_score": prng_score,
            "distance_entropy": distance_entropy,
            "distance_stats": distance_stats,
        }

    # Helper methods...
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        hist = np.bincount(data.flatten())
        hist = hist[hist > 0]
        hist = hist / len(data.flatten())
        return -np.sum(hist * np.log2(hist))

    def _chi_square_test(self, bits: np.ndarray) -> float:
        """Perform chi-square test for randomness"""
        observed = np.bincount(bits.flatten(), minlength=2)
        expected = np.array([len(bits.flatten()) / 2, len(bits.flatten()) / 2])
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        return 1 - p_value  # Convert to anomaly score

    def _runs_test(self, bits: np.ndarray) -> float:
        """Perform runs test for randomness"""
        runs = (
            len(
                [
                    i
                    for i in range(1, len(bits.flatten()))
                    if bits.flatten()[i] != bits.flatten()[i - 1]
                ]
            )
            + 1
        )
        n1 = np.sum(bits)
        n0 = len(bits.flatten()) - n1
        expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
        return 1 - abs(runs - expected_runs) / len(bits.flatten())

    def _autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation"""
        flat_data = data.flatten()
        return np.corrcoef(flat_data[:-lag], flat_data[lag:])[0, 1]

    def _calculate_density_map(
        self, lsbs: np.ndarray, window_size: int = 8
    ) -> np.ndarray:
        """
        Calculate local density map of LSB modifications
        """
        try:
            height, width = lsbs.shape
            density_map = np.zeros((height // window_size, width // window_size))

            for i in range(density_map.shape[0]):
                for j in range(density_map.shape[1]):
                    window = lsbs[
                        i * window_size : (i + 1) * window_size,
                        j * window_size : (j + 1) * window_size,
                    ]
                    density_map[i, j] = np.mean(window)

            return density_map

        except Exception as e:
            self.logger.error(f"Density map calculation failed: {str(e)}")
            return np.zeros((1, 1))

    def _detect_structured_patterns(
        self, positions: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """Detect if bit modifications follow structured patterns"""
        try:
            if len(positions[0]) < 2:
                return 0.0

            # Convert to coordinate pairs (ensure float type for calculations)
            coords = np.column_stack(
                [positions[0].astype(np.float64), positions[1].astype(np.float64)]
            )

            # Calculate distances between consecutive positions
            distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))

            # Analyze distance patterns
            distance_stats = {
                "mean": np.mean(distances),
                "std": np.std(distances),
                "entropy": self._calculate_entropy(distances),
            }

            # Score based on regularity
            regularity = 1.0 - (
                distance_stats["std"] / (distance_stats["mean"] + 1e-10)
            )
            entropy_score = 1.0 - (
                distance_stats["entropy"] / (np.log2(len(distances) + 1) + 1e-10)
            )

            return (regularity + entropy_score) / 2

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return 0.0

    def _analyze_bit_distribution(
        self, positions: Tuple[np.ndarray, np.ndarray]
    ) -> Dict:
        """
        Analyze the distribution of modified bits
        """
        try:
            if len(positions[0]) < 2:
                return {"uniformity": 0.0, "clustering": 0.0}

            coords = np.column_stack(positions)

            # Calculate spatial distribution metrics
            x_dist = np.histogram(coords[:, 0], bins="auto")[0]
            y_dist = np.histogram(coords[:, 1], bins="auto")[0]

            # Analyze uniformity
            x_uniformity = 1.0 - np.std(x_dist) / np.mean(x_dist)
            y_uniformity = 1.0 - np.std(y_dist) / np.mean(y_dist)

            # Analyze clustering

            clustering = DBSCAN(eps=3, min_samples=2).fit(coords)
            n_clusters = len(set(clustering.labels_)) - (
                1 if -1 in clustering.labels_ else 0
            )
            clustering_score = n_clusters / len(coords)

            return {
                "uniformity": (x_uniformity + y_uniformity) / 2,
                "clustering": clustering_score,
            }

        except Exception as e:
            self.logger.error(f"Bit distribution analysis failed: {str(e)}")
            return {"uniformity": 0.0, "clustering": 0.0}

    def _detect_encryption_signatures(self, bits: np.ndarray) -> Dict:
        """
        Look for signatures of known encryption methods
        """
        try:
            # Convert to bit sequence
            bit_sequence = bits.flatten()

            signatures = {
                "fernet": self._check_fernet_signature(bit_sequence),
                "aes": self._check_aes_signature(bit_sequence),
                "random_looking": self._check_randomness(bit_sequence),
            }

            return signatures

        except Exception as e:
            self.logger.error(f"Encryption signature detection failed: {str(e)}")
            return {"fernet": False, "aes": False, "random_looking": False}

    def _check_aes_signature(self, bits: np.ndarray) -> bool:
        """Check for AES encryption signatures"""
        try:
            if len(bits) < 128:  # AES block size
                return False

            # Look for block patterns
            blocks = bits[: len(bits) - (len(bits) % 128)].reshape(-1, 128)
            block_correlations = []

            for i in range(len(blocks) - 1):
                correlation = np.corrcoef(blocks[i], blocks[i + 1])[0, 1]
                block_correlations.append(abs(correlation))

            # AES typically shows very low block correlations
            return np.mean(block_correlations) < 0.1

        except Exception:
            return False

    def _check_randomness(self, bits: np.ndarray) -> bool:
        """Check if bit sequence appears random"""
        try:
            # Calculate runs test
            runs_score = self._runs_test(bits)

            # Calculate autocorrelation
            autocorr = self._autocorrelation(bits)

            # Calculate entropy
            entropy = self._calculate_entropy(bits)

            # Combine tests
            return runs_score > 0.8 and abs(autocorr) < 0.1 and entropy > 0.95

        except Exception:
            return False

    def _detect_prng_patterns(self, distances: np.ndarray) -> float:
        """
        Detect patterns consistent with PRNG-based positioning
        """
        try:
            # Calculate statistical properties
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            entropy = self._calculate_entropy(distances)

            # True random numbers should have high entropy
            # and consistent statistical properties
            entropy_score = entropy / np.log2(len(distances))
            consistency_score = 1.0 - (std_dist / mean_dist)

            return (entropy_score + consistency_score) / 2

        except Exception as e:
            self.logger.error(f"PRNG pattern detection failed: {str(e)}")
            return 0.0

    def _is_valid_length(self, length: int, total_size: int) -> bool:
        """Check if a potential message length is valid"""
        return 0 < length < total_size * 0.5

    def _assess_length_confidence(self, length: int, channel: np.ndarray) -> float:
        """Assess confidence in detected length field"""
        try:
            # Check if length makes sense for image size
            size_ratio = length / channel.size
            if size_ratio > 0.5 or size_ratio < 0.0001:
                return 0.0

            # Check if length correlates with modified bits
            lsbs = channel & 1
            modified_count = np.sum(lsbs)

            # Length should be less than modified bits
            if length > modified_count:
                return 0.0

            # Higher confidence if length is close to expected bit count
            ratio = length / modified_count
            return 1.0 - abs(0.5 - ratio)

        except Exception:
            return 0.0

    def _bits_to_int(self, bits: np.ndarray) -> int:
        """Convert bit array to integer"""
        try:
            # Ensure bits are integers
            bits = bits.astype(np.int8)
            return int("".join(map(str, bits)), 2)
        except Exception as e:
            self.logger.error(f"Bits to int conversion failed: {str(e)}")
            return 0

    def _check_fernet_signature(self, bits: np.ndarray) -> bool:
        """Check for Fernet encryption signatures"""
        try:
            # Fernet typically has a specific header pattern
            if len(bits) < 64:
                return False
            
            # Extract potential header
            header_bits = bits[:64]
            
            # Check for Fernet version (128 = 0x80)
            version_bits = header_bits[:8]
            version = self._bits_to_int(version_bits)
            
            # Check timestamp format (next 56 bits)
            timestamp_bits = header_bits[8:64]
            timestamp = self._bits_to_int(timestamp_bits)
            
            # Basic Fernet signature checks
            return (version == 128 and 
                    timestamp > 0 and 
                    timestamp < 2**64)
                
        except Exception as e:
            self.logger.error(f"Fernet signature check failed: {str(e)}")
            return False
