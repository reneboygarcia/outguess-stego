import numpy as np
from PIL import Image
import os
import base64
import hashlib
import random
from cryptography.fernet import Fernet
from ..utils.logger_config import get_logger
from .channel_analyzer import ChannelAnalyzer
from .error_correction import ErrorCorrection
from .quality_metrics import QualityAnalyzer

class ImageEncoder:
    """Class for encoding messages into images using steganography."""
    
    def __init__(self):
        self.logger = get_logger('image_encoder')
        self.channel_analyzer = ChannelAnalyzer()
        self.error_correction = ErrorCorrection()
        self.quality_analyzer = QualityAnalyzer()

    def encode(self, input_path: str, output_path: str, message: str, password: str, method: str = 'lsb') -> str:
        """Encode a message into an image using the specified method."""
        if method != 'lsb':
            raise ValueError(f"Unsupported encoding method: {method}")
            
        return self.hide_message(input_path, message, password, output_path)

    def hide_message(self, image_path: str, message: str, password: str, output_path: str) -> str:
        """Hide an encrypted message in an image"""
        try:
            # Generate encryption key and seed from password
            key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest()[:32])
            fernet = Fernet(key)
            seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
            
            # Select best channel and get image array
            selected_channel, pixels = self.channel_analyzer.select_best_channel(image_path)
            
            # Encrypt and add error correction
            encrypted_message = fernet.encrypt(message.encode())
            self.logger.debug(f"Encrypted message length: {len(encrypted_message)}")
            
            encoded_message = self.error_correction.encode_data(encrypted_message)
            message_length = len(encoded_message)
            self.logger.debug(f"Message length after ECC: {message_length}")
            
            # Store channel information in first byte
            channel_byte = selected_channel.value.to_bytes(1, byteorder='big')
            final_message = channel_byte + encoded_message
            message_length = len(final_message)
            
            # Get embedding positions
            positions = self._get_pixel_positions(pixels[:,:,0].shape, seed)
            
            # Prepare message bits
            length_bits = self._int_to_bits(message_length, 32)
            message_bits = self._bytes_to_bits(final_message)
            
            all_bits = length_bits + message_bits
            total_bits = len(all_bits)
            self.logger.debug(f"Total bits to embed: {total_bits}")
            
            # Check capacity
            if total_bits > len(positions):
                raise ValueError(f"Message too large. Need {total_bits} positions, have {len(positions)}")
            
            # Embed bits
            channel_data = pixels[:,:,selected_channel.value].copy()
            for i, bit in enumerate(all_bits):
                x, y = positions[i]
                channel_data[x, y] = self._embed_bit(channel_data[x, y], int(bit))
                self.logger.debug(f"Embedded bit at position ({x},{y})")
            
            pixels[:,:,selected_channel.value] = channel_data
            
            # Calculate quality metrics
            self._calculate_quality_metrics(pixels)
            
            # Save image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(pixels).save(output_path, format='PNG')
            self.logger.debug(f"Stego image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {str(e)}")
            raise

    def _get_pixel_positions(self, image_size: tuple, seed: int) -> list:
        """Generate pseudorandom pixel positions for embedding"""
        random.seed(seed)
        positions = [(x, y) for x in range(image_size[0]) for y in range(image_size[1])]
        random.shuffle(positions)
        return positions

    def _embed_bit(self, pixel: int, bit: int) -> int:
        """Embed a single bit into a pixel value"""
        return (pixel & 0xFE) | bit

    def _int_to_bits(self, value: int, bits: int) -> list:
        """Convert an integer to a list of bits"""
        return [int(b) for b in format(value, f'0{bits}b')]

    def _bytes_to_bits(self, data: bytes) -> list:
        """Convert bytes to a list of bits"""
        return [int(b) for byte in data for b in format(byte, '08b')]

    def _calculate_quality_metrics(self, pixels: np.ndarray) -> None:
        """Calculate and log quality metrics"""
        try:
            metrics = self.quality_analyzer.calculate_metrics(pixels.copy(), pixels)
            psnr = metrics['psnr']
            ssim = metrics['ssim']
            if np.isinf(psnr):
                self.logger.debug("PSNR is infinite, indicating identical images.")
            else:
                self.logger.debug(f"Quality metrics - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        except RuntimeWarning as e:
            self.logger.warning(f"Runtime warning during quality metrics calculation: {e}")
    