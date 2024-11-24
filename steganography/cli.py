#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Optional, Callable, TypeVar
from datetime import datetime
import questionary
from rich.console import Console
from rich.panel import Panel
from functools import wraps

from steganography.src.core.image_encoder import ImageEncoder
from steganography.src.core.image_decoder import ImageDecoder
from steganography.src.utils.logger_config import setup_logger

logger = setup_logger(__name__)
console = Console()

T = TypeVar('T')

def handle_back_option(func: Callable[..., Optional[T]]) -> Callable[..., Optional[T]]:
    """Decorator to handle 'back' option in user inputs."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[T]:
        result = func(*args, **kwargs)
        if isinstance(result, str) and result.lower() == 'back':
            return None
        return result
    return wrapper

def handle_errors(func: Callable) -> Callable:
    """Decorator to handle exceptions in command execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            console.print(f"❌ Error: {str(e)}", style="red")
            sys.exit(1)
    return wrapper

class SteganographyCLI:
    """Command Line Interface for steganography operations."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            description="Steganography Tool - Hide and extract messages from images",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--guided", action="store_true", help="Run in interactive guided mode"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        self._add_encode_parser(subparsers)
        self._add_decode_parser(subparsers)

        return parser

    def _add_encode_parser(self, subparsers):
        """Add encode command parser."""
        encode_parser = subparsers.add_parser("encode", help="Hide a message in an image")
        encode_parser.add_argument("-i", "--input", required=True, help="Path to input image")
        encode_parser.add_argument("-o", "--output", required=True, help="Path to output image")
        encode_parser.add_argument("-m", "--message", help="Message to hide")
        encode_parser.add_argument("-f", "--message-file", help="File containing message to hide")
        encode_parser.add_argument("-p", "--password", required=True, help="Password for encryption")
        encode_parser.add_argument("--method", choices=["lsb", "outguess"], required=True, help="Steganography method to use")

    def _add_decode_parser(self, subparsers):
        """Add decode command parser."""
        decode_parser = subparsers.add_parser("decode", help="Extract hidden message from an image")
        decode_parser.add_argument("-i", "--input", required=True, help="Path to image with hidden message")
        decode_parser.add_argument("-o", "--output", help="Path to save extracted message (optional)")
        decode_parser.add_argument("-p", "--password", required=True, help="Password for decryption")
        decode_parser.add_argument("--method", choices=["lsb", "outguess"], required=True, help="Steganography method to use")

    def _guided_mode(self) -> argparse.Namespace:
        """Interactive guided mode for user input."""
        console.print(Panel.fit("Welcome to the Steganography Tool!", title="🔒 Steganography Assistant"))

        while True:
            operation = self._select_operation()
            if operation == "Exit program":
                console.print("👋 Goodbye!", style="yellow")
                sys.exit(0)

            args = self._initialize_args(operation)

            while True:
                args.method = self._select_method()
                if args.method == "Back to main menu":
                    break

                if not self._get_common_inputs(args):
                    continue

                if args.command == "encode":
                    if not self._handle_encode_specific_inputs(args):
                        continue
                else:  # Decode
                    if not self._handle_decode_specific_inputs(args):
                        continue

                return args  # Return args if all inputs are valid

    def _select_operation(self) -> str:
        """Select operation from the menu."""
        return questionary.select(
            "What would you like to do?",
            choices=["Hide a message (encode)", "Extract a message (decode)", "Exit program"],
        ).ask()

    def _initialize_args(self, operation: str) -> argparse.Namespace:
        """Initialize command arguments based on selected operation."""
        args = argparse.Namespace()
        args.command = "encode" if "encode" in operation else "decode"
        return args

    def _select_method(self) -> str:
        """Select steganography method."""
        return questionary.select(
            "Select steganography method:",
            choices=["lsb", "Back to main menu"],
            default="lsb",
        ).ask()

    def _get_common_inputs(self, args: argparse.Namespace) -> bool:
        """Get common inputs for encoding/decoding."""
        args.input = self._get_input_path()
        if args.input is None:
            return False

        args.password = self._get_password()
        return args.password is not None

    @handle_back_option
    def _get_input_path(self) -> Optional[str]:
        """Prompt for input image path with validation."""
        return questionary.path(
            "Enter the path to your input image (or 'back' to return):",
            validate=lambda x: x.lower() == "back" or Path(x).exists(),
        ).ask()

    @handle_back_option
    def _get_password(self) -> Optional[str]:
        """Prompt for password with back option."""
        return questionary.password(
            "Enter the password for encryption/decryption (or 'back' to return):"
        ).ask()

    def _handle_encode_specific_inputs(self, args: argparse.Namespace) -> bool:
        """Handle inputs specific to encoding."""
        while True:
            args.output = self._get_output_path()
            if args.output is None:
                return False

            message_type = self._select_message_input_method()
            if message_type == "Back":
                continue

            if message_type == "Type directly":
                args.message = self._get_direct_message()
                if args.message is None:
                    continue
                args.message_file = None
            else:
                args.message_file = self._get_message_file_path()
                if args.message_file is None:
                    return False

            return True  # Proceed with encoding

    @handle_back_option
    def _get_output_path(self) -> Optional[str]:
        """Prompt for output image path with validation."""
        return questionary.path(
            "Enter the path for the output image (or 'back' to return):"
        ).ask()

    def _select_message_input_method(self) -> str:
        """Select method for inputting the message."""
        return questionary.select(
            "How would you like to input your message?",
            choices=["Type directly", "From a file", "Back"],
        ).ask()

    @handle_back_option
    def _get_direct_message(self) -> Optional[str]:
        """Prompt for direct message input."""
        return questionary.text(
            "Enter your secret message (or 'back' to return):"
        ).ask()

    def _get_message_file_path(self) -> Optional[str]:
        """Prompt for message file path with validation."""
        return questionary.path(
            "Enter the path to your message file (or 'back' to return):",
            validate=lambda x: x.lower() == "back" or Path(x).exists(),
        ).ask()

    def _handle_decode_specific_inputs(self, args: argparse.Namespace) -> bool:
        """Handle inputs specific to decoding."""
        save_output = questionary.confirm(
            "Would you like to save the extracted message to a file?", default=False
        ).ask()

        if save_output:
            args.output = self._get_output_specification()
            if args.output is None:
                return False

        return True  # Proceed with decoding

    def _get_output_specification(self) -> Optional[str]:
        """Prompt for output specification."""
        output_type = questionary.select(
            "How would you like to specify the output?",
            choices=["Specify directory (filename will be generated)", "Specify complete filename", "Back"],
        ).ask()

        if output_type == "Back":
            return None

        if "directory" in output_type:
            return questionary.path(
                "Enter the directory to save the message (or 'back' to return):",
                only_directories=True,
            ).ask()
        else:
            return questionary.path(
                "Enter the complete path for the output file (or 'back' to return):"
            ).ask()

    def _validate_paths(self, input_path: str, output_path: str) -> tuple[Path, Path]:
        """Validate input and output paths."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return input_path, output_path

    def _get_message(self, message: Optional[str], message_file: Optional[str]) -> str:
        """Get message from direct input or file."""
        if not message and not message_file:
            raise ValueError("No message provided. Use --message or --message-file")

        if message and message_file:
            raise ValueError("Please provide either message or message-file, not both")

        if message_file:
            return self._read_message_from_file(message_file)

        return message

    def _read_message_from_file(self, message_file: str) -> str:
        """Read message from a file."""
        try:
            with open(message_file, "r") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read message file: {e}")

    @handle_errors
    def encode(self, args) -> None:
        """Handle encode command."""
        input_path, output_path = self._validate_paths(args.input, args.output)
        message = self._get_message(args.message, args.message_file)

        with console.status("Encoding message..."):
            encoder = ImageEncoder()
            encoder.encode(
                input_path=str(input_path),
                output_path=str(output_path),
                message=message,
                password=args.password,
                method=args.method,
            )

        console.print(f"✅ Message successfully encoded to {output_path}", style="green")

    @handle_errors
    def decode(self, args) -> None:
        """Handle decode command."""
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with console.status("Decoding message..."):
            decoder = ImageDecoder()
            message = decoder.decode(
                input_path=str(input_path),
                password=args.password,
                method=args.method,
            )

        if args.output:
            self._save_extracted_message(message, args.output)
        else:
            console.print(Panel.fit(message, title="📝 Extracted Message", border_style="green"))

    def _save_extracted_message(self, message: str, output_path: str) -> None:
        """Save the extracted message to a specified output path."""
        output_path = Path(output_path)

        if output_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / f"extracted_message_{timestamp}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(message)
        console.print(f"✅ Extracted message saved to {output_path}", style="green")

    def run(self) -> None:
        """Execute the CLI application."""
        try:
            args = self.parser.parse_args()
            args = self._handle_guided_mode(args)

            self._execute_command(args)
        except KeyboardInterrupt:
            self._handle_interruption()

    def _handle_guided_mode(self, args) -> Optional[argparse.Namespace]:
        """Handle guided mode input."""
        if args.guided:
            args = self._guided_mode()
            if not args:  # User chose to exit
                return None
        return args

    def _execute_command(self, args) -> None:
        """Execute the specified command."""
        command_actions = {
            "encode": self.encode,
            "decode": self.decode,
        }
        action = command_actions.get(args.command)
        if action:
            action(args)
        else:
            self.parser.print_help()
            sys.exit(1)

    def _handle_interruption(self) -> None:
        """Handle keyboard interruption gracefully."""
        console.print("\n👋 Operation cancelled by user", style="yellow")
        sys.exit(0)


def main():
    """Entry point for the CLI application."""
    try:
        cli = SteganographyCLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n👋 Operation cancelled by user", style="yellow")
        sys.exit(0)


if __name__ == "__main__":
    main()
