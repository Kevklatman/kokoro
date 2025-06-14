#!/usr/bin/env python3
"""
Interactive CLI for generating audio using Kokoro TTS service.
Can connect to either a local instance or a Cloud Run deployment.
Uses only standard library modules.
"""

import argparse
import os
import sys
import json
import base64
import http.client
import urllib.parse
import ssl
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import subprocess
import time

# Constants
DEFAULT_LOCAL_HOST = "localhost"
DEFAULT_LOCAL_PORT = 8080
DEFAULT_CLOUD_HOST = "apiforswifttts-696551753574.europe-west1.run.app"

def play_audio(audio_data: bytes) -> None:
    """Play audio using the system's default audio player"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(audio_data)
    
    try:
        console.print(f"[green]Playing audio from {temp_file_path}...[/green]")
        
        # Cross-platform audio playing
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", temp_file_path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(temp_file_path)  # This will open with the default program
        else:  # Linux and others
            subprocess.run(["xdg-open", temp_file_path], check=True)
    except Exception as e:
        console.print(f"[red]Error playing audio: {str(e)}[/red]")
    finally:
        # Clean up the temp file after a delay to ensure it can be played
        time.sleep(1)  # Give time for the player to open the file
        try:
            os.remove(temp_file_path)
        except:
            pass

def save_audio(audio_data: bytes, filename: str) -> None:
    """Save audio to a file"""
    with open(filename, "wb") as f:
        f.write(audio_data)
    console.print(f"[green]Audio saved to {filename}[/green]")

def get_available_voices(api_url: str) -> List[str]:
    """Get list of available voices from the API"""
    try:
        response = requests.get(f"{api_url}/voices", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            # Handle case where API returns a different structure
            if isinstance(data, dict) and "voices" in data:
                return data["voices"]
        console.print(f"[yellow]Couldn't get voices from API: {response.status_code}[/yellow]")
    except requests.RequestException as e:
        console.print(f"[red]Error connecting to API: {str(e)}[/red]")
    
    # Return a default list if we can't get from API
    return ["en_US_1", "en_US_2", "en_GB_1"]

def generate_audio(
    api_url: str,
    text: str,
    voice: str,
    quality: str = "medium",
    format: str = "mp3",
    save_path: Optional[str] = None,
    play: bool = False
) -> Optional[bytes]:
    """Generate audio from text using the TTS API"""
    
    console.print(f"[blue]Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'[/blue]")
    console.print(f"[blue]Using voice: {voice}, quality: {quality}, format: {format}[/blue]")
    
    # Prepare the request
    payload = {
        "text": text,
        "voice": voice,
        "quality": quality,
        "format": format
    }
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating audio...", total=None)
            
            # Make the API request
            response = requests.post(
                f"{api_url}/tts", 
                json=payload,
                timeout=60  # TTS can take a while for long texts
            )
            
            progress.update(task, completed=True, total=1)
            
            if response.status_code != 200:
                console.print(f"[red]Error: API returned status code {response.status_code}[/red]")
                console.print(f"[red]Response: {response.text}[/red]")
                return None
            
            # Handle the response based on content type
            if "audio/" in response.headers.get("Content-Type", ""):
                audio_data = response.content
                console.print(f"[green]Successfully generated audio ({len(audio_data)/1024:.1f} KB)[/green]")
                
                # Save the audio if requested
                if save_path:
                    save_audio(audio_data, save_path)
                
                # Play the audio if requested
                if play:
                    play_audio(audio_data)
                
                return audio_data
            elif response.headers.get("Content-Type") == "application/json":
                # Handle case where response might be base64-encoded
                try:
                    data = response.json()
                    if "audio" in data and isinstance(data["audio"], str):
                        # Assume base64 encoding
                        audio_data = base64.b64decode(data["audio"])
                        console.print(f"[green]Successfully decoded base64 audio ({len(audio_data)/1024:.1f} KB)[/green]")
                        
                        if save_path:
                            save_audio(audio_data, save_path)
                        
                        if play:
                            play_audio(audio_data)
                        
                        return audio_data
                except Exception as e:
                    console.print(f"[red]Error decoding JSON response: {str(e)}[/red]")
            
            console.print(f"[red]Unexpected response format: {response.headers.get('Content-Type')}[/red]")
            return None
            
    except requests.RequestException as e:
        console.print(f"[red]Error connecting to API: {str(e)}[/red]")
        return None

def interactive_mode(api_url: str) -> None:
    """Run an interactive CLI session"""
    console.print(f"[bold green]Kokoro TTS Interactive CLI[/bold green]")
    console.print(f"[bold blue]Connected to: {api_url}[/bold blue]")

    # Get available voices
    voices = get_available_voices(api_url)
    console.print(f"[bold]Available voices:[/bold] {', '.join(voices) or 'Unknown (using defaults)'}")

    while True:
        console.rule("[bold]New TTS Request[/bold]")
        
        # Get user input
        text = Prompt.ask("[bold]Enter text to convert to speech[/bold] (or 'q' to quit)")
        if text.lower() == 'q':
            break
        
        # Select voice
        voice_options = voices if voices else ["en_US_1", "en_US_2", "en_GB_1"]
        voice_idx = 0 if not voice_options else Prompt.ask(
            "[bold]Select voice[/bold]", 
            choices=[str(i) for i in range(len(voice_options))],
            default="0"
        )
        voice = voice_options[int(voice_idx)]
        
        # Select quality
        quality = Prompt.ask(
            "[bold]Select quality[/bold]", 
            choices=["high", "medium", "low"], 
            default="medium"
        )
        
        # Select format
        format = Prompt.ask(
            "[bold]Select format[/bold]", 
            choices=["mp3", "wav"], 
            default="mp3"
        )
        
        # Choose whether to save
        save = Confirm.ask("[bold]Save to file?[/bold]", default=False)
        save_path = None
        if save:
            save_path = Prompt.ask(
                "[bold]Enter save path[/bold]", 
                default=f"output_{int(time.time())}.{format}"
            )
        
        # Choose whether to play
        play = Confirm.ask("[bold]Play audio?[/bold]", default=True)
        
        # Generate the audio
        generate_audio(
            api_url=api_url,
            text=text,
            voice=voice,
            quality=quality,
            format=format,
            save_path=save_path,
            play=play
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Kokoro TTS CLI")
    
    # Setup connection options
    connection_group = parser.add_argument_group("Connection Options")
    connection_group.add_argument(
        "--local", 
        action="store_true",
        help="Connect to a local instance (default: http://localhost:8080)"
    )
    connection_group.add_argument(
        "--cloud", 
        action="store_true",
        help="Connect to Cloud Run instance"
    )
    connection_group.add_argument(
        "--url", 
        type=str,
        help="Specify a custom API URL"
    )
    
    # Direct generation options
    direct_group = parser.add_argument_group("Direct Generation Options")
    direct_group.add_argument(
        "--text", 
        type=str,
        help="Text to convert to speech (enables direct mode)"
    )
    direct_group.add_argument(
        "--voice", 
        type=str,
        help="Voice to use for TTS"
    )
    direct_group.add_argument(
        "--quality", 
        choices=["high", "medium", "low"],
        default="medium",
        help="Audio quality (default: medium)"
    )
    direct_group.add_argument(
        "--format", 
        choices=["mp3", "wav"],
        default="mp3",
        help="Audio format (default: mp3)"
    )
    direct_group.add_argument(
        "--output", 
        type=str,
        help="Output file path (default: output_<timestamp>.mp3)"
    )
    direct_group.add_argument(
        "--play",
        action="store_true",
        help="Play the audio after generation"
    )
    
    args = parser.parse_args()
    
    # Determine API URL
    if args.url:
        api_url = args.url
    elif args.cloud:
        api_url = DEFAULT_CLOUD_URL
    else:
        # Default to local
        api_url = DEFAULT_LOCAL_URL
    
    # Check if we're in direct mode or interactive mode
    if args.text:
        # Direct mode
        voice = args.voice or "en_US_1"  # Default voice
        output_path = args.output or f"output_{int(time.time())}.{args.format}"
        
        generate_audio(
            api_url=api_url,
            text=args.text,
            voice=voice,
            quality=args.quality,
            format=args.format,
            save_path=output_path,
            play=args.play
        )
    else:
        # Interactive mode
        try:
            interactive_mode(api_url)
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            sys.exit(0)

if __name__ == "__main__":
    main()
