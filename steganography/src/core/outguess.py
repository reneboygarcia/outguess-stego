import numpy as np
from PIL import Image
import random
import hashlib
import os
from cryptography.fernet import Fernet
import logging
import base64
from ..utils.logger_config import get_logger
from .channel_analyzer import ChannelAnalyzer, Channel
from .error_correction import ErrorCorrection
from .quality_metrics import QualityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OutguessStego:
    def __init__(self):
        self.logger = get_logger('outguess')
        self.seed = None
        self.capacity = 0
        self.channel_analyzer = ChannelAnalyzer()
        self.error_correction = ErrorCorrection()
        self.quality_analyzer = QualityAnalyzer()
        self.selected_channel = None

    def _get_pixel_positions(self, image_size: tuple, seed: int) -> list:
        """Generate pseudorandom pixel positions for embedding"""
        random.seed(seed)
        positions = [(x, y) for x in range(image_size[0]) for y in range(image_size[1])]
        random.shuffle(positions)
        return positions

    def _can_embed(self, pixel_value: int) -> bool:
        """Check if pixel is suitable for embedding"""
        return 20 < pixel_value < 235  # Avoid extreme values

    def _embed_bit(self, pixel: int, bit: int) -> int:
        """Embed a single bit into a pixel value"""
        return (pixel & 0xFE) | bit

    def _extract_bit(self, pixel: int) -> int:
        """Extract a single bit from a pixel value"""
        return pixel & 1

    def hide_message(self, image_path: str, message: str, password: str) -> str:
        """Hide an encrypted message in an image"""
        # Generate encryption key and seed
        key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest()[:32])
        fernet = Fernet(key)
        self.seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
        
        # Select best channel and get image array
        selected_channel, pixels = self.channel_analyzer.select_best_channel(image_path)
        self.selected_channel = selected_channel
        
        # Encrypt and add error correction
        encrypted_message = fernet.encrypt(message.encode())
        self.logger.debug(f"Encrypted message length: {len(encrypted_message)}")
        
        encoded_message = self.error_correction.encode_data(encrypted_message)
        message_length = len(encoded_message)
        self.logger.debug(f"Message length after ECC: {message_length}")
        
        # Get embedding positions
        positions = self._get_pixel_positions(pixels[:,:,0].shape, self.seed)
        
        # Prepare message bits
        length_bytes = message_length.to_bytes(4, byteorder='big')
        length_bits = []
        for byte in length_bytes:
            length_bits.extend(format(byte, '08b'))
        
        message_bits = []
        for byte in encoded_message:
            message_bits.extend(format(byte, '08b'))
        
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
        metrics = self.quality_analyzer.calculate_metrics(pixels.copy(), pixels)
        self.logger.debug(f"Quality metrics - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
        
        # Save image
        output_path = os.path.join(
            os.path.dirname(image_path), 
            "stego_" + os.path.splitext(os.path.basename(image_path))[0] + ".png"
        )
        Image.fromarray(pixels).save(output_path, format='PNG')
        self.logger.debug(f"Stego image saved to: {output_path}")
        return output_path

    def extract_message(self, image_path: str, password: str) -> str:
        """Extract and decrypt a hidden message from an image"""
        try:
            # Load image
            with Image.open(image_path) as img:
                pixels = np.array(img)
                self.logger.debug(f"Image loaded with shape: {pixels.shape}")
                
                # Use blue channel by default
                channel_data = pixels[:,:,Channel.BLUE.value]
                self.logger.debug(f"Using BLUE channel")
            
            # Generate key and seed
            key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest()[:32])
            fernet = Fernet(key)
            self.seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
            
            # Get positions
            positions = self._get_pixel_positions(channel_data.shape, self.seed)
            
            # Extract length (32 bits)
            length_bits = []
            for i in range(32):
                x, y = positions[i]
                bit = self._extract_bit(channel_data[x, y])
                length_bits.append(str(bit))
                self.logger.debug(f"Extracted length bit {i}: {bit}")
            
            # Convert length bits to integer
            length_binary = ''.join(length_bits)
            message_length = int(length_binary, 2)
            self.logger.debug(f"Extracted message length: {message_length}")
            
            if message_length <= 0 or message_length > 1000:  # Reasonable limit
                raise ValueError(f"Invalid message length: {message_length}")
            
            # Extract message bits
            message_bits = []
            for i in range(message_length * 8):
                x, y = positions[i + 32]
                bit = self._extract_bit(channel_data[x, y])
                message_bits.append(str(bit))
                self.logger.debug(f"Extracted message bit {i}: {bit}")
            
            # Convert bits to bytes
            message_binary = ''.join(message_bits)
            message_bytes = bytearray()
            for i in range(0, len(message_binary), 8):
                byte = message_binary[i:i+8]
                message_bytes.append(int(byte, 2))
            
            # Decrypt and return
            decrypted_message = fernet.decrypt(bytes(message_bytes))
            return decrypted_message.decode()
            
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error args: {e.args}")
            raise ValueError(f"Failed to extract message: {str(e)}")
