#!/usr/bin/env python3
"""
Simple TTS CLI for generating audio using Kokoro TTS service.
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
from typing import Optional, Dict, Any
import tempfile
import subprocess
import time

#

# Constants
DEFAULT_LOCAL_HOST = "localhost"
DEFAULT_LOCAL_PORT = 8080
DEFAULT_CLOUD_HOST = "apiforswifttts-696551753574.europe-west1.run.app"

def print_colored(text, color=None):
    """Print colored text if supported"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "bold": "\033[1m",
        "end": "\033[0m"
    }
    
    if color and sys.stdout.isatty():  # Check if terminal supports colors
        print(f"{colors.get(color, '')}{text}{colors['end']}")
    else:
        print(text)

def make_http_request(host: str, path: str, method: str = "GET", 
                     data: Optional[Dict] = None, use_ssl: bool = False,
                     headers: Optional[Dict] = None, max_redirects: int = 5,
                     auth_token: Optional[str] = None) -> tuple:
    """Make an HTTP request and return status code and response data"""
    if headers is None:
        headers = {"Content-Type": "application/json"}
    
    # Add authorization if token is provided
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    redirect_count = 0
    original_host = host
    original_path = path
    
    while redirect_count < max_redirects:
        # Set up connection based on current host and SSL setting
        current_ssl = use_ssl
        if "://" in host:
            if host.startswith("https://"):
                current_ssl = True
                host = host.replace("https://", "")
            elif host.startswith("http://"):
                current_ssl = False
                host = host.replace("http://", "")
        
        if current_ssl:
            conn = http.client.HTTPSConnection(host)
        else:
            conn = http.client.HTTPConnection(host)
        
        # Convert data to JSON if needed
        body = json.dumps(data) if data else None
        
        try:
            # Make request
            print_colored(f"Connecting to {'https://' if current_ssl else 'http://'}{host}{path}", "blue")
            
            # Debug request info if verbosity enabled
            if os.environ.get("TTS_DEBUG"):
                print_colored("--- DEBUG: Request Details ---", "yellow")
                print(f"Method: {method}")
                print(f"Headers: {headers}")
                if body:
                    print(f"Body: {body[:200]}{'...' if len(body) > 200 else ''}")
            
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            
            status = response.status
            
            # Handle redirects
            if status in (301, 302, 303, 307, 308):
                redirect_url = response.getheader('Location')
                if not redirect_url:
                    return status, None  # No redirect URL provided
                
                print_colored(f"Following redirect ({status}) to: {redirect_url}", "yellow")
                redirect_count += 1
                
                # Parse the redirect URL
                if "://" in redirect_url:
                    # Absolute URL
                    parsed = urllib.parse.urlparse(redirect_url)
                    host = parsed.netloc
                    path = parsed.path
                    if parsed.query:
                        path += f"?{parsed.query}"
                    use_ssl = parsed.scheme == "https"
                else:
                    # Relative URL
                    path = redirect_url
                
                conn.close()
                continue  # Try again with new URL
            
            # Read response data for non-redirect responses
            response_data = response.read()
            
            # Try to parse as JSON if applicable
            content_type = response.getheader("Content-Type", "")
            if "application/json" in content_type:
                try:
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    pass
            
            return status, response_data
        
        except Exception as e:
            print_colored(f"Error making request: {str(e)}", "red")
            return None, None
        finally:
            conn.close()
    
    # If we get here, we've exceeded the maximum redirects
    print_colored(f"Error: Exceeded maximum redirects ({max_redirects})", "red")
    return None, None

def play_audio(audio_data: bytes, format="mp3") -> None:
    """Play audio using system's default player"""
    # Use the correct file extension based on the audio format
    suffix = f".{format.lower()}"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(audio_data)
    
    try:
        print_colored(f"Playing audio from {temp_file_path}...", "green")
        
        # Cross-platform audio playing
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", temp_file_path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(temp_file_path)  # This will open with the default program
        else:  # Linux and others
            subprocess.run(["xdg-open", temp_file_path], check=True)
    except Exception as e:
        print_colored(f"Error playing audio: {str(e)}", "red")
    finally:
        # Clean up temp file after a delay
        time.sleep(1)  # Give time for the player to open the file
        try:
            os.remove(temp_file_path)
        except:
            pass

def save_audio(audio_data: bytes, filename: str) -> None:
    """Save audio to file"""
    # Ensure the audio_tests directory exists
    audio_tests_dir = "audio_tests"
    os.makedirs(audio_tests_dir, exist_ok=True)
    # If filename is not already in audio_tests, prepend it
    if not filename.startswith(audio_tests_dir + os.sep):
        filename = os.path.join(audio_tests_dir, filename)
    with open(filename, "wb") as f:
        f.write(audio_data)
    print_colored(f"Audio saved to {filename}", "green")

def get_available_voices(host: str, use_ssl: bool = False, auth_token: Optional[str] = None) -> list:
    """Get available voice options from the API"""
    status, data = make_http_request(
        host=host, 
        path="/voices", 
        method="GET", 
        use_ssl=use_ssl,
        auth_token=auth_token
    )
    
    # Always ensure af_sky and af_heart are first in the list
    default_voices = ["af_sky", "af_heart"]
    voices = []
    if status == 200 and data:
        # Handle different response formats
        if isinstance(data, list):
            voices = data
        elif isinstance(data, dict) and "voices" in data:
            voices = data["voices"]
    else:
        print_colored("Could not retrieve voices from API, using default set", "yellow")
        voices = ["en_US_1", "en_US_2", "en_GB_1"]
    # Prepend af_sky and af_heart if not present
    for v in reversed(default_voices):
        if v in voices:
            voices.remove(v)
        voices.insert(0, v)
    return voices

def generate_audio(
    host: str, 
    text: str, 
    voice: str = "en_US_1",
    quality: str = "medium",
    format: str = "mp3",
    save_path: Optional[str] = None,
    play: bool = False,
    use_ssl: bool = False,
    auth_token: Optional[str] = None,
    cloud_mode: bool = False
) -> Optional[bytes]:
    """Generate audio from text using the TTS API"""
    
    print_colored(f"Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'", "blue")
    print_colored(f"Using voice: {voice}, quality: {quality}, format: {format}", "blue")
    
    # Prepare request data based on cloud mode or local mode
    if cloud_mode:
        # Cloud endpoint may use a different structure
        payload = {
            "text": text,
            "voice": voice,
            "quality": quality,
            "output_format": format,  # Cloud might use different parameter names
            "return_format": "base64"  # Ensure we get base64 back
        }
        # Use the /jobs/submit endpoint that we've seen in logs
        api_path = "/jobs/submit"
    else:
        # Local endpoint uses the standard format
        payload = {
            "text": text,
            "voice": voice,
            "quality": quality,
            "format": format
        }
        api_path = "/tts"
    
    print_colored("Sending request to server...", "blue")
    status, response = make_http_request(
        host=host,
        path=api_path,
        method="POST",
        data=payload,
        use_ssl=use_ssl,
        auth_token=auth_token
    )
    
    if not status or status != 200:
        print_colored(f"Error: Received status code {status}", "red")
        
        # Try to extract error details if possible
        error_msg = "Unknown error"
        if isinstance(response, dict) and "error" in response:
            error_msg = response["error"]
        elif isinstance(response, dict) and "message" in response:
            error_msg = response["message"]
        elif isinstance(response, bytes):
            try:
                # Try to decode as text for error messages
                error_text = response.decode('utf-8', errors='ignore')[:500]
                print_colored(f"Response content: {error_text}", "red")
            except:
                pass
        
        print_colored(f"Error details: {error_msg}", "red")
        
        # Suggest solutions based on status code
        if status == 500:
            print_colored("This could be a server-side issue. Try the following:", "yellow")
            print_colored("1. Check if your authentication token is valid", "yellow")
            print_colored("2. Try a different voice or shorter text", "yellow")
            print_colored("3. Verify the cloud service is running correctly", "yellow")
            print_colored("4. Run with TTS_DEBUG=1 for more detailed logs", "yellow")
        elif status == 401 or status == 403:
            print_colored("Authentication error - check your token", "yellow")
        
        return None
    
    audio_data = None
    
    # Handle the response based on content type
    if isinstance(response, bytes):
        # Direct binary response
        audio_data = response
    elif isinstance(response, dict) and "audio" in response:
        # Base64-encoded response
        try:
            audio_data = base64.b64decode(response["audio"])
        except Exception as e:
            print_colored(f"Error decoding base64 audio: {str(e)}", "red")
            return None
    
    if audio_data:
        print_colored(f"Successfully generated audio ({len(audio_data)/1024:.1f} KB)", "green")
        
        # Save if requested
        if save_path:
            save_audio(audio_data, save_path)
        
        # Play if requested
        if play:
            play_audio(audio_data, format=format)
        
        return audio_data
    else:
        print_colored("Error: Could not extract audio data from response", "red")
        return None

def input_with_default(prompt: str, default: str = "") -> str:
    """Get input with a default value"""
    result = input(f"{prompt} [{default}]: ") if default else input(f"{prompt}: ")
    return result if result else default

def confirm(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation"""
    default_txt = "Y/n" if default else "y/N"
    result = input(f"{prompt} [{default_txt}]: ").lower()
    
    if not result:
        return default
    
    return result in ["y", "yes"]

def interactive_mode(host: str, use_ssl: bool, auth_token: Optional[str] = None, cloud_mode: bool = False) -> None:
    """Run interactive mode"""
    print_colored("=== Kokoro TTS Interactive CLI ===", "bold")
    print_colored(f"Connected to: {'https://' if use_ssl else 'http://'}{host}", "blue")
    
    # Get available voices
    print_colored("Fetching available voices...", "blue")
    voices = get_available_voices(host, use_ssl, auth_token)
    print_colored(f"Available voices: {', '.join(voices)}", "green")
    
    while True:
        print_colored("\n=== New TTS Request ===", "bold")
        
        # Get text
        text = input_with_default("Enter text to convert to speech (or 'q' to quit)")
        if text.lower() == 'q':
            break
        
        # Voice selection
        print_colored("\nAvailable voices:", "blue")
        for i, v in enumerate(voices):
            print(f"{i}: {v}")
        
        voice_idx_str = input_with_default("Select voice number", "0")
        try:
            voice_idx = int(voice_idx_str)
            voice = voices[voice_idx] if 0 <= voice_idx < len(voices) else voices[0]
        except (ValueError, IndexError):
            print_colored("Invalid selection, using first voice", "yellow")
            voice = voices[0]
        
        # Quality selection
        print_colored("\nQuality options:", "blue")
        print("0: high")
        print("1: medium")
        print("2: low")
        
        quality_idx_str = input_with_default("Select quality", "1")
        quality_options = ["high", "medium", "low"]
        try:
            quality_idx = int(quality_idx_str)
            quality = quality_options[quality_idx] if 0 <= quality_idx < len(quality_options) else "medium"
        except (ValueError, IndexError):
            print_colored("Invalid selection, using medium quality", "yellow")
            quality = "medium"
        
        # Format selection
        format = "mp3"  # Default
        if confirm("Use MP3 format? (WAV otherwise)", True):
            format = "mp3"
        else:
            format = "wav"
        
        # Save option
        save = confirm("Save to file?", False)
        save_path = None
        if save:
            default_filename = f"output_{int(time.time())}.{format}"
            # Prepend audio_tests dir to default
            default_filename = os.path.join("audio_tests", default_filename)
            save_path = input_with_default("Enter filename", default_filename)
        
        # Play option
        play = confirm("Play audio when ready?", True)
        
        # Generate audio
        print_colored("\nGenerating audio...", "blue")
        generate_audio(
            host=host,
            text=text,
            voice=voice,
            quality=quality,
            format=format,
            save_path=save_path,
            play=play,
            use_ssl=use_ssl,
            auth_token=auth_token,
            cloud_mode=cloud_mode
        )

def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple TTS CLI using standard library")
    
    # Connection options
    connection_group = parser.add_argument_group("Connection Options")
    connection_group.add_argument(
        "--local", 
        action="store_true",
        help=f"Connect to local instance (default: {DEFAULT_LOCAL_HOST}:{DEFAULT_LOCAL_PORT})"
    )
    connection_group.add_argument(
        "--cloud", 
        action="store_true",
        help=f"Connect to Cloud Run ({DEFAULT_CLOUD_HOST})"
    )
    connection_group.add_argument(
        "--token",
        type=str,
        help="Authentication token for cloud deployment"
    )
    connection_group.add_argument(
        "--cloud-jobs-api",
        action="store_true",
        help="Use the cloud jobs API format instead of the standard API format"
    )
    connection_group.add_argument(
        "--host",
        type=str,
        help="Custom host to connect to"
    )
    connection_group.add_argument(
        "--port",
        type=int,
        help="Custom port to use with host"
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
        default="en_US_1",
        help="Voice to use for TTS (default: en_US_1)"
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
        help="Output file path"
    )
    direct_group.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation"
    )
    
    args = parser.parse_args()
    
    # Determine connection details
    host = args.host
    port = args.port
    use_ssl = False
    cloud_mode = args.cloud_jobs_api or args.cloud  # Use cloud format with either flag
    
    if not host:
        if args.cloud:
            host = DEFAULT_CLOUD_HOST
            use_ssl = True
        else:  # Default to local
            host = DEFAULT_LOCAL_HOST
            port = DEFAULT_LOCAL_PORT
    
    # Include port in host if specified
    if port and not args.cloud:
        host = f"{host}:{port}"
    
    # Determine mode (direct or interactive)
    if args.text:
        # Direct mode
        # Save to audio_tests by default
        output_path = args.output or os.path.join("audio_tests", f"output_{int(time.time())}.{args.format}")
        # Use the --voice argument if provided, otherwise default to af_sky
        voice = args.voice if args.voice is not None else "af_sky"
        generate_audio(
            host=host,
            text=args.text,
            voice=voice,
            quality=args.quality,
            format=args.format,
            save_path=output_path,
            play=args.play,
            use_ssl=use_ssl,
            auth_token=args.token,
            cloud_mode=cloud_mode
        )
    else:
        # Interactive mode
        try:
            interactive_mode(host, use_ssl, args.token, cloud_mode)
        except KeyboardInterrupt:
            print_colored("\nExiting...", "yellow")
            sys.exit(0)

if __name__ == "__main__":
    main()
