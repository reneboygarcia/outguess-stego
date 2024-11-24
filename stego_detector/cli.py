#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, Sequence, Callable, Any, List
from functools import wraps
from datetime import datetime
import questionary
from stego_detector.detect_steganography import StegoCracker
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle exceptions in CLI methods"""

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        try:
            return func(self, *args, **kwargs)
        except KeyboardInterrupt:
            self.console.print("\n👋 Operation cancelled by user", style="yellow")
            return self.handle_navigation()
        except Exception as e:
            self.logger.exception("An error occurred")
            self.console.print(f"❌ Error: {str(e)}", style="red")
            return self.handle_navigation()

    return wrapper


def handle_back_option(
    func: Callable[..., Optional[Any]]
) -> Callable[..., Optional[Any]]:
    """Decorator to handle 'back' option in user inputs."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Optional[Any]:
        result = func(*args, **kwargs)
        if isinstance(result, str) and result.lower() == "back":
            return None
        return result

    return wrapper


def require_confirmation(message: str) -> Callable:
    """Decorator to require user confirmation before proceeding"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if questionary.confirm(message, default=True).ask():
                return func(self, *args, **kwargs)
            return self.handle_navigation()

        return wrapper

    return decorator


class RichHelpFormatter(argparse.HelpFormatter):
    """Custom formatter for rich text in help messages"""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar = action.metavar or default
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = action.metavar or default
                for option_string in action.option_strings:
                    parts.append(f"[cyan]{option_string}[/cyan]")
                parts[-1] += f" {args_string}"
            return ", ".join(parts)


class StegoDetectorCLI:
    """Command-line interface for the steganography detector"""

    def __init__(self):
        self.console = Console()
        self.setup_logging()
        self.cracker = StegoCracker()
        self.current_results = None
        self.current_image_path = None

    def setup_logging(self) -> None:
        """Configure logging with rich formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        self.logger = logging.getLogger("stego_detector.cli")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(
            description="[bold blue]F00D-grade steganography detector[/bold blue]",
            formatter_class=RichHelpFormatter,
        )

        parser.add_argument(
            "--guided", action="store_true", help="Run in interactive guided mode"
        )

        parser.add_argument(
            "image_path", type=str, nargs="?", help="Path to the image file to analyze"
        )

        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "-o", "--output", type=str, help="Path to save analysis results"
        )

        parser.add_argument(
            "--quick",
            action="store_true",
            help="Perform quick analysis (less thorough)",
        )

        return parser

    def _guided_mode(self) -> Optional[argparse.Namespace]:
        """Interactive guided mode with improved navigation"""
        self.console.print(
            Panel.fit(
                "Welcome to the F00D-Grade Steganography Detector!",
                title="🔍 Stego Detector Assistant",
            )
        )

        while True:
            operation = questionary.select(
                "What would you like to do?",
                choices=[
                    "Analyze new image",
                    "View last results",
                    "Save results",
                    "Back to main menu",
                    "Exit program",
                ],
            ).ask()

            match operation:
                case "Exit program":
                    if self._confirm_exit():
                        self.console.print("👋 Goodbye!", style="yellow")
                        sys.exit(0)
                    continue

                case "Analyze new image":
                    image_path = self._get_image_path()
                    if image_path:
                        return self._create_analysis_args(image_path)
                    continue

                case "View last results":
                    self.show_last_results()
                    continue

                case "Save results":
                    self.save_results_prompt()
                    continue

                case "Back to main menu":
                    self._handle_back()
                    return None

    @handle_back_option
    def _get_image_path(self) -> Optional[str]:
        """Prompt for image path with improved navigation"""
        while True:
            # First ask for navigation choice
            choice = questionary.select(
                "How would you like to select the image?",
                choices=[
                    "Enter image path",
                    "Browse recent files",
                    "Browse Downloads folder",
                    "Browse Desktop folder",
                    "Browse custom location",
                    "Back to main menu",
                ],
            ).ask()

            if choice == "Back to main menu":
                self._handle_back()
                return None

            try:
                if choice == "Enter image path":
                    # Use questionary.path for path input
                    image_path = questionary.path(
                        "Enter the path to your image:",
                        validate=lambda x: Path(x).exists()
                        and Path(x).suffix.lower()
                        in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"},
                    ).ask()

                    if image_path:
                        return image_path
                    continue

                elif choice == "Browse recent files":
                    # Get recently accessed image files
                    recent_files = self._get_recent_image_files()
                    if not recent_files:
                        self.console.print(
                            "[yellow]No recent image files found[/yellow]"
                        )
                        continue

                    file_choice = questionary.select(
                        "Select a recent image file:",
                        choices=["Back"] + [str(f) for f in recent_files],
                    ).ask()

                    if file_choice == "Back":
                        continue
                    return file_choice

                elif choice == "Browse Downloads folder":
                    downloads_path = self._get_downloads_folder()
                    return self._browse_directory(downloads_path)

                elif choice == "Browse Desktop folder":
                    desktop_path = Path.home() / "Desktop"
                    return self._browse_directory(desktop_path)

                else:  # Browse custom location
                    return self._browse_directory(Path.home())

            except Exception as e:
                self.logger.error(f"Error browsing files: {str(e)}")
                self.console.print("[red]Error browsing files. Please try again.[/red]")
                continue

    def _get_recent_image_files(self, max_files: int = 10) -> List[Path]:
        """Get list of recently accessed image files"""
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
        recent_files = []

        # Check common locations
        locations = [
            self._get_downloads_folder(),
            Path.home() / "Desktop",
            Path.home() / "Pictures",
        ]

        for location in locations:
            if location.exists():
                for ext in image_extensions:
                    recent_files.extend(location.glob(f"*{ext}"))

        # Sort by last modified time
        recent_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return recent_files[:max_files]

    def _browse_directory(self, start_path: Path) -> Optional[str]:
        """Browse directory for image files"""
        current_path = start_path
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        while True:
            # Get directories and image files
            try:
                items = list(current_path.iterdir())
                dirs = [item for item in items if item.is_dir()]
                files = [
                    item
                    for item in items
                    if item.is_file() and item.suffix.lower() in image_extensions
                ]

                # Prepare choices
                choices = [
                    "📁 .." if current_path != Path.home() else "📁 Home",
                    *[f"📁 {d.name}" for d in sorted(dirs)],
                    *[f"🖼 {f.name}" for f in sorted(files)],
                    "Back to navigation",
                ]

                # Show current path
                self.console.print(f"\n[blue]Current location: {current_path}[/blue]")

                choice = questionary.select(
                    "Select a file or directory:", choices=choices
                ).ask()

                if choice == "Back to navigation":
                    return None

                if choice.startswith("📁 .."):
                    current_path = current_path.parent
                    continue

                if choice.startswith("📁 Home"):
                    current_path = Path.home()
                    continue

                if choice.startswith("📁 "):
                    current_path = current_path / choice[2:]
                    continue

                if choice.startswith("🖼 "):
                    return str(current_path / choice[2:])

            except Exception as e:
                self.logger.error(f"Error browsing directory: {str(e)}")
                self.console.print(
                    "[red]Error accessing directory. Please try again.[/red]"
                )
                return None

    def _create_analysis_args(self, image_path: str) -> argparse.Namespace:
        """Create argument namespace for analysis."""
        args = argparse.Namespace()
        args.image_path = image_path
        args.verbose = questionary.confirm(
            "Enable verbose output?", default=False
        ).ask()

        if questionary.confirm(
            "Would you like to save the analysis results?", default=True
        ).ask():
            args.output = self._get_output_path()
        else:
            args.output = None

        args.quick = questionary.confirm(
            "Perform quick analysis (less thorough)?", default=False
        ).ask()

        return args

    def _get_downloads_folder(self) -> Path:
        """Get the default downloads folder based on OS"""
        if sys.platform == "win32":
            import winreg

            sub_key = (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            )
            downloads_guid = "{374DE290-123F-4565-9164-39C4925E467B}"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                location = winreg.QueryValueEx(key, downloads_guid)[0]
            return Path(location)
        else:  # macOS and Linux
            return Path.home() / "Downloads"

    @handle_back_option
    def _get_output_path(self, existing_path: Optional[Path] = None) -> Optional[str]:
        """Prompt for output path specification."""
        # Get default downloads folder
        downloads_dir = self._get_downloads_folder()
        default_output_dir = downloads_dir / "stego_analysis"

        if existing_path:
            # Handle existing path logic
            default_output_dir.mkdir(parents=True, exist_ok=True)

            if existing_path.is_dir() or not existing_path.suffix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stego_analysis_{timestamp}.txt"
                return str(
                    default_output_dir / filename
                    if not existing_path.is_dir()
                    else existing_path / filename
                )
            return str(existing_path)

        # Interactive path selection
        output_type = questionary.select(
            "How would you like to specify the output?",
            choices=[
                "Use default location (Downloads)",
                "Specify directory",
                "Specify complete filename",
                "Back",
            ],
        ).ask()

        match output_type:
            case "Back":
                return None

            case "Use default location (Downloads)":
                default_output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return str(default_output_dir / f"stego_analysis_{timestamp}.txt")

            case _ if "directory" in output_type:
                directory = questionary.path(
                    "Enter the directory to save the results:", only_directories=True
                ).ask()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                return str(Path(directory) / f"stego_analysis_{timestamp}.txt")

            case _:
                return questionary.path(
                    "Enter the complete path for the output file:"
                ).ask()

    @handle_errors
    def analyze_image(self, image_path: Path) -> int:
        """Analyze an image file"""
        if not image_path.exists():
            self.console.print(
                f"❌ [red]Error: Image file not found: {image_path}[/red]"
            )
            return self.handle_navigation()

        self.display_header()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("🔍 Analyzing image...", total=None)
                self.current_results = self.cracker.crack_image(str(image_path))

                if not self.current_results:
                    self.console.print(
                        "[yellow]⚠️ Analysis completed but no results were generated[/yellow]"
                    )
                    return self.handle_navigation()

                self.current_image_path = image_path
                progress.update(task, completed=True)

            if self.current_results:
                self.display_results(self.current_results)

            return self.handle_navigation()

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.console.print(f"❌ [red]Error during analysis: {str(e)}[/red]")
            return self.handle_navigation()

    def display_header(self) -> None:
        """Display stylized header"""
        header = Panel(
            "[bold blue]NSA-Grade Steganography Detector[/bold blue]\n"
            "[dim]Detecting hidden messages in images[/dim]",
            style="bold white",
        )
        self.console.print(header)

    # def display_menu(self) -> None:
    #     """Display main menu options"""
    #     menu = Panel(
    #         "[bold green]1.[/bold green] 🔍 Analyze new image\n"
    #         "[bold green]2.[/bold green] 📄 View last results\n"
    #         "[bold green]3.[/bold green] 💾 Save results\n"
    #         "[bold green]4.[/bold green] 🚪 Exit",
    #         title="[bold]Menu Options[/bold]",
    #         style="bold white",
    #     )
    #     self.console.print(menu)

    def handle_navigation(self) -> int:
        """Handle menu navigation with improved back/exit handling"""
        while True:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "🔍 Analyze new image",
                    "📄 View last results",
                    "💾 Save results",
                    "↩ Back to main menu",
                    "🚪 Exit program",
                ],
            ).ask()

            match choice:
                case "🔍 Analyze new image":
                    image_path = self._get_image_path()
                    if image_path:
                        return self.analyze_image(Path(image_path))
                    continue

                case "📄 View last results":
                    self.show_last_results()
                    continue

                case "💾 Save results":
                    self.save_results_prompt()
                    continue

                case "↩ Back to main menu":
                    self._handle_back()
                    return self.run()

                case "🚪 Exit program":
                    if self._confirm_exit():
                        self.console.print("👋 Goodbye!", style="yellow")
                        return 0
                    continue

    def exit_program(self) -> int:
        """Exit the program with confirmation"""
        if self._confirm_exit():
            self.console.print("👋 Goodbye!", style="yellow")
            return 0
        return self.handle_navigation()

    @handle_errors
    def prompt_for_analysis(self) -> int:
        """Prompt user for image analysis"""
        image_path = input("Enter the path to the image file: ")
        if not image_path:
            return self.handle_navigation()

        return self.analyze_image(Path(image_path))

    @handle_errors
    def show_last_results(self) -> int:
        """Display results from last analysis"""
        if self.current_results is None:
            self.console.print("[yellow]No analysis results available[/yellow]")
        else:
            self.display_results(self.current_results)
        return self.handle_navigation()

    @handle_errors
    @require_confirmation("Do you want to save the results?")
    def save_results_prompt(self) -> int:
        """Prompt user to save results"""
        if self.current_results is None:
            self.console.print("[yellow]No results available to save[/yellow]")
            return self.handle_navigation()

        # Get default downloads folder based on OS
        if sys.platform == "win32":
            downloads_dir = Path.home() / "Downloads"
        else:  # macOS and Linux
            downloads_dir = Path.home() / "Downloads"

        # Create default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"stego_analysis_{timestamp}.txt"
        default_path = downloads_dir / default_filename

        # Prompt user with default path
        self.console.print(f"Default save location: {default_path}")
        output_path = input("Press Enter to use default or enter custom path: ").strip()

        if not output_path:
            output_path = default_path

        self.save_results(Path(output_path))
        return self.handle_navigation()

    def display_results(self, results: Optional[dict]) -> None:
        """Display analysis results in a formatted table"""
        if not results:
            self.console.print("[yellow]⚠️ No results available to display[/yellow]")
            return

        table = Table(title="Analysis Results", show_header=True)

        table.add_column("Analysis Type", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Confidence", style="yellow")

        try:
            # Standard Analysis
            standard_results = results.get("standard")
            if standard_results is not None:
                verdict = getattr(standard_results, "verdict", None)
                if verdict:
                    verdict_text, confidence = verdict
                else:
                    verdict_text, confidence = "Unknown", "N/A"

                table.add_row("Standard Analysis", verdict_text, confidence)

                embedding_technique = getattr(
                    standard_results, "embedding_technique", "Unknown"
                )
                overall_probability = getattr(
                    standard_results, "overall_probability", 0.0
                )
                table.add_row(
                    "Embedding Technique",
                    str(embedding_technique),
                    f"{float(overall_probability):.2f}",
                )

            # Channel Analysis
            meta_conf = results.get("meta") or {}
            confidence_data = meta_conf.get("confidence") or {}
            channel_conf = confidence_data.get("channel_confidence")

            if channel_conf is not None:
                crack_results = results.get("crack") or {}
                channel_info = crack_results.get("channel") or {}
                likely_channel = channel_info.get("likely_channel", "Unknown")
                table.add_row(
                    "Channel Analysis",
                    f"Channel {likely_channel}",
                    f"{float(channel_conf):.2f}",
                )

            # Encryption Analysis
            advanced_results = results.get("advanced") or {}
            if advanced_results and advanced_results.get("encryption"):
                enc_conf = confidence_data.get("encryption_confidence", 0.0)
                table.add_row(
                    "Encryption Analysis",
                    "Detected" if enc_conf > 0.7 else "Not Detected",
                    f"{float(enc_conf):.2f}",
                )

            # Overall Assessment
            overall_conf = confidence_data.get("overall_confidence", 0.0)
            table.add_row(
                "[bold]Overall Assessment[/bold]",
                (
                    "[bold]Hidden Content Present[/bold]"
                    if overall_conf > 0.7
                    else "[bold]No Hidden Content[/bold]"
                ),
                f"[bold]{float(overall_conf):.2f}[/bold]",
            )

            self.console.print(table)

        except Exception as e:
            self.logger.error(f"Error displaying results: {str(e)}")
            self.console.print(
                "[red]❌ Error displaying results. See logs for details.[/red]"
            )

    @handle_errors
    def save_results(self, output_path: Path) -> None:
        """Save analysis results to file"""
        if not self.current_results:
            self.console.print("[yellow]No results available to save[/yellow]")
            return

        try:
            # Get the final path using the helper method
            final_path = self._get_output_path(output_path)
            if not final_path:
                return

            # Create parent directory only if user has explicitly chosen a path
            if final_path != output_path:
                Path(final_path).parent.mkdir(parents=True, exist_ok=True)

            # Write results
            with open(final_path, "w") as f:
                self.cracker._write_detailed_results(f, self.current_results)

            self.console.print(
                f"✅ Detailed results saved to: {final_path}", style="green"
            )

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            self.console.print(f"❌ Error saving results: {str(e)}", style="red")

    def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure the specified directory exists"""
        directory.mkdir(parents=True, exist_ok=True)

    @handle_errors
    def run(self, args: Optional[Sequence[str]] = None) -> int:
        """Execute the CLI application"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if parsed_args.guided or not parsed_args.image_path:
            guided_args = self._guided_mode()
            if guided_args:
                return self.analyze_image(Path(guided_args.image_path))
            return 0

        if parsed_args.image_path:
            return self.analyze_image(Path(parsed_args.image_path))

        self.display_header()
        return self.handle_navigation()

    def _confirm_exit(self) -> bool:
        """Confirm before exiting"""
        if self.current_results:
            return questionary.confirm(
                "❓ You have unsaved results. Are you sure you want to exit?",
                default=False,
            ).ask()
        return True

    def _handle_back(self) -> None:
        """Handle back navigation"""
        self.console.print("[blue]↩ Going back...[/blue]")


def main() -> int:
    """Entry point for the CLI application"""
    try:
        cli = StegoDetectorCLI()
        return cli.run()
    except KeyboardInterrupt:
        Console().print("\n👋 Operation cancelled by user", style="yellow")
        return 130


if __name__ == "__main__":
    sys.exit(main())
