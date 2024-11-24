import numpy as np
from PIL import Image
import base64
import random
import hashlib
from cryptography.fernet import Fernet
from ..utils.logger_config import get_logger
from .channel_analyzer import Channel
from .error_correction import ErrorCorrection

class ImageDecoder:
    """Class for decoding messages from steganographic images."""
    
    def __init__(self):
        self.logger = get_logger('image_decoder')
        self.error_correction = ErrorCorrection()

    def decode(self, input_path: str, password: str, method: str = 'lsb') -> str:
        """Decode a message from an image using the specified method."""
        if method != 'lsb':
            raise ValueError(f"Unsupported decoding method: {method}")
            
        return self.extract_message(input_path, password)

    def extract_message(self, image_path: str, password: str) -> str:
        """Extract and decrypt a hidden message from an image"""
        try:
            # Load image
            with Image.open(image_path) as img:
                pixels = np.array(img)
                self.logger.debug(f"Image loaded with shape: {pixels.shape}")
            
            # Generate key and seed from password
            key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest()[:32])
            fernet = Fernet(key)
            seed = int(hashlib.sha256(password.encode()).hexdigest(), 16)
            
            # Get positions
            positions = self._get_pixel_positions(pixels[:,:,0].shape, seed)
            
            # Extract length (32 bits)
            length_bits = self._extract_bits(pixels, positions, 0, 32)
            message_length = int(''.join(map(str, length_bits)), 2)
            self.logger.debug(f"Extracted message length: {message_length}")
            
            if message_length <= 0 or message_length > 1000:  # Reasonable limit
                raise ValueError(f"Invalid message length: {message_length}")
            
            # Extract message bits
            message_bits = self._extract_bits(pixels, positions, 32, message_length * 8)
            
            # Convert bits to bytes
            message_bytes = self._bits_to_bytes(message_bits)
            
            # First byte contains channel information
            channel_value = message_bytes[0]
            channel = Channel(channel_value)
            encoded_message = bytes(message_bytes[1:])  # Convert to bytes explicitly
            
            # Apply error correction decoding
            try:
                decoded_message = self.error_correction.decode_data(encoded_message)
            except Exception as e:
                self.logger.error(f"Error correction decoding failed: {e}")
                decoded_message = encoded_message
            
            # Ensure decoded_message is bytes
            if not isinstance(decoded_message, bytes):
                decoded_message = bytes(decoded_message)
            
            # Decrypt and return
            try:
                decrypted_message = fernet.decrypt(decoded_message)
                return decrypted_message.decode('utf-8')
            except Exception as e:
                self.logger.error(f"Decryption failed: {e}")
                raise ValueError("Failed to decrypt message. Invalid password or corrupted data.")
            
        except Exception as e:
            self.logger.error(f"Error during extraction: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error args: {e.args}")
            raise ValueError(f"Failed to extract message: {str(e)}")

    def _get_pixel_positions(self, image_size: tuple, seed: int) -> list:
        """Generate pseudorandom pixel positions for embedding"""
        random.seed(seed)
        positions = [(x, y) for x in range(image_size[0]) for y in range(image_size[1])]
        random.shuffle(positions)
        return positions

    def _extract_bit(self, pixel: int) -> int:
        """Extract a single bit from a pixel value"""
        return pixel & 1

    def _extract_bits(self, pixels: np.ndarray, positions: list, start: int, count: int) -> list:
        """Extract a sequence of bits from the image"""
        bits = []
        for i in range(count):
            x, y = positions[start + i]
            bit = self._extract_bit(pixels[x, y, Channel.BLUE.value])
            bits.append(bit)
            self.logger.debug(f"Extracted bit {start + i}: {bit}")
        return bits

    def _bits_to_bytes(self, bits: list) -> bytes:
        """Convert a list of bits to bytes"""
        bit_string = ''.join(map(str, bits))
        # Ensure the bit string length is a multiple of 8
        if len(bit_string) % 8:
            bit_string = bit_string.ljust((len(bit_string) + 7) & ~7, '0')
        
        # Convert to bytes
        bytes_list = []
        for i in range(0, len(bit_string), 8):
            byte = int(bit_string[i:i+8], 2)
            bytes_list.append(byte)
        
        return bytes(bytes_list)

def main():
    image_path = input("Enter the path to the stego image file: ")
    password = input("Enter the password for decryption: ")

    decoder = ImageDecoder()  # Create an instance of ImageDecoder
    try:
        message = decoder.extract_message(image_path, password)  # Call the method on the instance
        print(f"Extracted message: {message}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 