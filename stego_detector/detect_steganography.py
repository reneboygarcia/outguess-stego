import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, Tuple
import logging
from PIL import Image
from datetime import datetime

# Add parent directory to path to import stego_detector
sys.path.append(str(Path(__file__).parent.parent))

from stego_detector.detector import StegoDetector
from stego_detector.analyzers.reverse_pattern import ReversePatternAnalyzer
from stego_detector.crackers.encoder_cracker import EncoderCracker
from stego_detector.crackers.advanced_cracker import AdvancedCracker
from stego_detector.utils.image_data import ImageData


class StegoCracker:
    """Food-grade steganography cracking tool"""

    def __init__(self):
        # Configure logging
        self.logger = self._setup_logger()

        # Initialize analyzers and crackers
        self.detector = StegoDetector()
        self.reverse_analyzer = ReversePatternAnalyzer()
        self.encoder_cracker = EncoderCracker()
        self.advanced_cracker = AdvancedCracker()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("stego_cracker")
        logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler()
        log_file_path = "steganography-project/stego_detector/stego_analysis.log"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)

        # Create formatters and add it to handlers
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setFormatter(logging.Formatter(log_format))

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def crack_image(self, image_path: str) -> Dict:
        """Attempt to crack steganographic content using multiple approaches"""
        self.logger.info(f"Starting analysis on: {image_path}")
        print("\n[+] Starting Food-grade stego analysis...")

        results = {
            "standard": None,
            "reverse": None,
            "crack": None,
            "advanced": None,
            "meta": {
                "confidence": {
                    "channel_confidence": 0.0,
                    "encryption_confidence": 0.0,
                    "overall_confidence": 0.0,
                }
            },
        }

        try:
            # Load and convert image properly
            image = Image.open(image_path)
            # Create ImageData object
            image_data = ImageData(
                raw_data=np.array(image, dtype=np.uint8),
                color_space="RGB" if image.mode == "RGB" else image.mode,
                format=image.format,
            )

            analysis_methods = {
                "standard": (
                    self.detector.analyze,
                    image_path,
                    "Running standard stego detection...",
                ),
                "reverse": (
                    self.reverse_analyzer.analyze,
                    image_data,
                    "Running reverse pattern analysis...",
                ),
                "crack": (
                    self.encoder_cracker.crack,
                    image_data,
                    "Running encoder cracking analysis...",
                ),
                "advanced": (
                    self.advanced_cracker.analyze,
                    image_data,
                    "Running advanced cracking analysis...",
                ),
            }

            for key, (method, data, message) in analysis_methods.items():
                try:
                    self.logger.info(message)
                    results[key] = method(data)
                except Exception as e:
                    self.logger.error(f"{message} failed: {str(e)}")

            # Only combine results if we have some successful analyses
            if any(v is not None for v in [results["standard"], results["advanced"]]):
                results = self._combine_results(
                    results["standard"],
                    results["reverse"],
                    results["crack"],
                    results["advanced"],
                )

            # Display results
            self._display_forensic_results(results)

            # Save analysis artifacts (but don't force save)
            # Let the CLI handle the save location
            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return results

    def _combine_results(
        self, standard_result, reverse_results, crack_results, advanced_results
    ) -> Dict:
        """Combine and correlate results from different analyzers"""
        combined = {
            "standard": standard_result,
            "reverse": reverse_results,
            "crack": crack_results,
            "advanced": advanced_results,
            "meta": {
                "confidence": {
                    "channel_confidence": 0.0,
                    "encryption_confidence": 0.0,
                    "overall_confidence": 0.0,
                }
            },
        }

        try:
            if crack_results or advanced_results:
                combined["meta"]["confidence"].update(
                    {
                        "channel_confidence": self._calculate_channel_confidence(
                            crack_results or {}, advanced_results or {}
                        ),
                        "encryption_confidence": self._calculate_encryption_confidence(
                            standard_result, advanced_results or {}
                        ),
                        "overall_confidence": self._calculate_overall_confidence(
                            standard_result, advanced_results
                        ),
                    }
                )
        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {str(e)}")

        return combined

    def _calculate_channel_confidence(
        self, crack_results: Dict, advanced_results: Dict
    ) -> float:
        """Calculate confidence in channel detection"""
        try:
            # Extract channel info and confidence scores
            crack_info = crack_results.get("channel", {})
            advanced_info = advanced_results.get("channels", {})
            
            crack_channel = crack_info.get("likely_channel")
            crack_confidence = crack_info.get("confidence", 0.0)
            
            advanced_channel = advanced_info.get("most_likely") 
            advanced_confidence = advanced_info.get("confidence", 0.0)

            # No channels detected
            if not crack_channel and not advanced_channel:
                return 0.0

            # Single channel detected
            if not crack_channel:
                return advanced_confidence * 0.5
            if not advanced_channel:
                return crack_confidence * 0.5

            # Both channels detected
            if crack_channel == advanced_channel:
                return min(crack_confidence + advanced_confidence, 1.0)
            
            return max(crack_confidence, advanced_confidence) * 0.5

        except Exception as e:
            self.logger.error(f"Error calculating channel confidence: {str(e)}")
            return 0.0

    def _calculate_encryption_confidence(
        self, standard_result, advanced_results
    ) -> float:
        """Calculate confidence in encryption detection"""
        standard_sig = standard_result.encryption_signature
        advanced_sig = advanced_results.get("encryption", {}).get("encryption_score", 0)

        return (standard_sig + advanced_sig) / 2

    def _calculate_overall_confidence(self, standard_result, advanced_results) -> float:
        """Calculate overall confidence in detection"""
        standard_prob = standard_result.overall_probability
        advanced_prob = advanced_results.get("encryption", {}).get(
            "likely_encrypted", 0
        )

        return (standard_prob + advanced_prob) / 2

    def _display_forensic_results(self, results: Dict):
        """Display comprehensive forensic analysis results"""
        print("\n[+] Forensic Analysis Results:")
        print("=" * 60)

        # Standard Analysis
        if results.get("standard"):
            print("\nStandard Analysis:")
            print(f"Overall probability: {results['standard'].overall_probability:.3f}")
            verdict, confidence = results["standard"].verdict
            print(f"Verdict: {verdict} (Confidence: {confidence})")
            if hasattr(results["standard"], "embedding_technique"):
                print(f"Likely technique: {results['standard'].embedding_technique}")

        # Channel Analysis
        if results.get("meta", {}).get("confidence", {}).get("channel_confidence"):
            print("\nChannel Analysis:")
            channel_conf = results["meta"]["confidence"]["channel_confidence"]
            print(f"Channel detection confidence: {channel_conf:.3f}")
            if channel_conf > 0.7 and results.get("crack", {}).get("channel"):
                crack_channel = results["crack"]["channel"]["likely_channel"]
                print(f"Most likely embedding channel: {crack_channel}")

        # Advanced Analysis
        if results.get("advanced"):
            if "encryption" in results["advanced"]:
                print("\nEncryption Analysis:")
                enc = results["advanced"]["encryption"]
                if "randomness_tests" in enc:
                    print("Randomness tests:")
                    for test, value in enc["randomness_tests"].items():
                        print(f"- {test}: {value:.3f}")

            if "password_patterns" in results["advanced"]:
                print("\nPattern Analysis:")
                pp = results["advanced"]["password_patterns"]
                if "prng_score" in pp:
                    print(f"Password-based probability: {pp['prng_score']:.3f}")
                if "distance_stats" in pp:
                    print("Distance statistics:")
                    for stat, value in pp["distance_stats"].items():
                        print(f"- {stat}: {value:.3f}")

        # Overall Assessment
        print("\n[+] Overall Assessment:")
        print("=" * 60)
        overall_conf = (
            results.get("meta", {}).get("confidence", {}).get("overall_confidence", 0)
        )
        print(f"Overall confidence: {overall_conf:.3f}")

        # Recommendations
        self._display_recommendations(results)

    def _display_recommendations(self, results: Dict):
        """Display attack recommendations based on analysis"""
        print("\n[+] Recommendations:")
        print("=" * 60)

        if results["meta"]["confidence"]["overall_confidence"] > 0.7:
            if (
                results["advanced"]
                .get("password_patterns", {})
                .get("password_based", False)
            ):
                print("- Strong indication of password-based hiding")
                print("- Recommend password recovery attack")
                print("- Consider analyzing bit distribution patterns")

            if results["advanced"].get("encryption", {}).get("likely_encrypted", False):
                print("- Encrypted content detected")
                print("- Consider cryptographic analysis")
                print("- Look for encryption implementation weaknesses")
        else:
            print("- No strong indicators of steganographic content")
            print("- Consider deeper statistical analysis")
            print("- Verify image integrity")

    def _get_downloads_folder(self) -> Path:
        """Get the default downloads folder based on OS"""
        if sys.platform == "win32":
            import winreg
            sub_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            downloads_guid = "{374DE290-123F-4565-9164-39C4925E467B}"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                location = winreg.QueryValueEx(key, downloads_guid)[0]
            return Path(location)
        else:  # macOS and Linux
            return Path.home() / "Downloads"

    def _save_analysis_artifacts(self, image_path: str, results: Dict):
        """Save analysis artifacts and visualizations"""
        # Use Downloads folder as default location
        downloads_dir = self._get_downloads_folder()
        output_dir = downloads_dir / "stego_analysis"
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"{Path(image_path).stem}_analysis_{timestamp}.txt"

        with open(result_file, "w") as f:
            self._write_detailed_results(f, results)

        self.logger.info(f"Analysis artifacts saved to: {result_file}")
        return result_file  # Return the path for reference

    def _write_detailed_results(self, file, results: Dict):
        """Write detailed analysis results to file"""
        try:
            file.write("Steganography Analysis Detailed Results\n")
            file.write("=" * 50 + "\n\n")

            # Standard Analysis
            if results.get("standard"):
                file.write("Standard Analysis:\n")
                file.write("-" * 20 + "\n")
                file.write(
                    f"Overall probability: {results['standard'].overall_probability:.3f}\n"
                )
                verdict, confidence = results["standard"].verdict
                file.write(f"Verdict: {verdict} (Confidence: {confidence})\n")
                if hasattr(results["standard"], "embedding_technique"):
                    file.write(
                        f"Likely technique: {results['standard'].embedding_technique}\n"
                    )
                file.write("\n")

            # Channel Analysis
            if results.get("meta", {}).get("confidence", {}).get("channel_confidence"):
                file.write("Channel Analysis:\n")
                file.write("-" * 20 + "\n")
                channel_conf = results["meta"]["confidence"]["channel_confidence"]
                file.write(f"Channel detection confidence: {channel_conf:.3f}\n")
                if channel_conf > 0.7 and results.get("crack", {}).get("channel"):
                    crack_channel = results["crack"]["channel"]["likely_channel"]
                    file.write(f"Most likely embedding channel: {crack_channel}\n")
                file.write("\n")

            # Advanced Analysis
            if results.get("advanced"):
                file.write("Advanced Analysis:\n")
                file.write("-" * 20 + "\n")

                if "encryption" in results["advanced"]:
                    file.write("Encryption Analysis:\n")
                    enc = results["advanced"]["encryption"]
                    if "randomness_tests" in enc:
                        file.write("Randomness tests:\n")
                        for test, value in enc["randomness_tests"].items():
                            file.write(f"- {test}: {value:.3f}\n")
                    file.write("\n")

                if "password_patterns" in results["advanced"]:
                    file.write("Pattern Analysis:\n")
                    pp = results["advanced"]["password_patterns"]
                    if "prng_score" in pp:
                        file.write(
                            f"Password-based probability: {pp['prng_score']:.3f}\n"
                        )
                    if "distance_stats" in pp:
                        file.write("Distance statistics:\n")
                        for stat, value in pp["distance_stats"].items():
                            file.write(f"- {stat}: {value:.3f}\n")
                    file.write("\n")

            # Overall Assessment
            file.write("Overall Assessment:\n")
            file.write("-" * 20 + "\n")
            overall_conf = (
                results.get("meta", {})
                .get("confidence", {})
                .get("overall_confidence", 0)
            )
            file.write(f"Overall confidence: {overall_conf:.3f}\n\n")

            # Raw Data (for debugging)
            file.write("\nRaw Analysis Data:\n")
            file.write("-" * 20 + "\n")
            for key, value in results.items():
                if key != "meta":
                    file.write(f"\n{key.upper()}:\n")
                    file.write(str(value))
                    file.write("\n")

        except Exception as e:
            self.logger.error(f"Error writing detailed results: {str(e)}")


def main():
    cracker = StegoCracker()
    image_path = input("Enter the path to the image to analyze: ")
    cracker.crack_image(image_path)


if __name__ == "__main__":
    main()
