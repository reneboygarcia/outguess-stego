from reedsolo import RSCodec
import logging

class ErrorCorrection:
    def __init__(self, ecc_symbols=32):
        """Initialize Reed-Solomon codec with specified number of error correction symbols"""
        self.rsc = RSCodec(ecc_symbols)
        self.ecc_symbols = ecc_symbols
        
    def encode_data(self, data: bytes) -> bytes:
        """Add error correction codes to data"""
        try:
            encoded_data = self.rsc.encode(data)
            logging.debug(f"Added {self.ecc_symbols} ECC symbols to {len(data)} bytes of data")
            return encoded_data
        except Exception as e:
            logging.error(f"Error during ECC encoding: {str(e)}")
            raise ValueError(f"Failed to encode data with ECC: {str(e)}")
    
    def decode_data(self, data: bytes) -> bytes:
        """Decode and correct data using error correction codes"""
        try:
            # The decode method returns (decoded_msg, decoded_msg_with_ecc, syndromes)
            # We only need the decoded message
            decoded_data = self.rsc.decode(data)[0]
            logging.debug(f"Successfully decoded {len(data)} bytes to {len(decoded_data)} bytes")
            return decoded_data
        except Exception as e:
            logging.error(f"Error during ECC decoding: {str(e)}")
            raise ValueError(f"Failed to decode data with ECC: {str(e)}") 