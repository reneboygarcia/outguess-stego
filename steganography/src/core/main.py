from steganography.src.core.outguess import OutguessStego

def main():
    stego = OutguessStego()
    
    # Get user input for image path, message, and password
    image_path = input("Enter the path to the image file: ")
    message = input("Enter the message to hide: ")
    password = input("Enter the password: ")
    
    # Hide a message
    try:
        output_path = stego.hide_message(image_path, message, password)
        print(f"Message hidden in {output_path}")
        
        # Extract the message
        extracted_message = stego.extract_message(output_path, password)
        print(f"Extracted message: {extracted_message}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 