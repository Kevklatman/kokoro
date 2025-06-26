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
DEFAULT_LOCAL_URL = f"http://{DEFAULT_LOCAL_HOST}:{DEFAULT_LOCAL_PORT}"
DEFAULT_CLOUD_URL = f"https://{DEFAULT_CLOUD_HOST}"

def play_audio(audio_data: bytes) -> None:
    """Play audio using the system's default audio player"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(audio_data)
    
    try:
        print(f"Playing audio from {temp_file_path}...")
        
        # Cross-platform audio playing
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", temp_file_path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(temp_file_path)  # This will open with the default program
        else:  # Linux and others
            subprocess.run(["xdg-open", temp_file_path], check=True)
    except Exception as e:
        print(f"Error playing audio: {str(e)}")
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
    print(f"Audio saved to {filename}")

def make_http_request(url: str, method: str = "GET", headers: Dict = None, body: Dict = None, timeout: int = 10) -> Tuple[int, Any]:
    """Make an HTTP request using standard library and return status code and response data"""
    parsed_url = urllib.parse.urlparse(url)
    is_https = parsed_url.scheme == 'https'
    host = parsed_url.netloc
    path = parsed_url.path
    
    # Add query parameters if present
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"
    
    if headers is None:
        headers = {}
    
    # Set default headers
    if method in ["POST", "PUT", "PATCH"] and body is not None:
        headers["Content-Type"] = "application/json"
    
    try:
        if is_https:
            conn = http.client.HTTPSConnection(host, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(host, timeout=timeout)
        
        # Convert body to JSON if present
        json_body = None
        if body is not None:
            json_body = json.dumps(body).encode('utf-8')
        
        conn.request(method, path, json_body, headers)
        response = conn.getresponse()
        
        # Get status code and response body
        status_code = response.status
        
        # Handle redirects manually
        if status_code in (301, 302, 303, 307, 308):
            location = response.getheader('Location')
            if location:
                print(f"Following redirect to {location}")
                # If it's a relative URL, construct the full URL
                if location.startswith('/'):
                    redirect_url = f"{parsed_url.scheme}://{host}{location}"
                else:
                    redirect_url = location
                # Follow the redirect with same method for 307/308, GET for others
                if status_code in (307, 308):
                    return make_http_request(redirect_url, method, headers, body, timeout)
                else:
                    return make_http_request(redirect_url, "GET", headers, None, timeout)
        
        # Read the response data
        response_data = response.read()
        
        # Parse JSON response if possible
        if response_data:
            try:
                return status_code, json.loads(response_data.decode('utf-8'))
            except json.JSONDecodeError:
                return status_code, response_data
        
        return status_code, None
    
    except Exception as e:
        print(f"Error making HTTP request: {str(e)}")
        return 500, None
    finally:
        if 'conn' in locals():
            conn.close()

def get_available_voices(api_url: str) -> List[str]:
    """Get list of available voices from the API"""
    # Our custom Kokoro voices - these should always be available
    kokoro_voices = ["af_sky", "af_heart"]
    
    print(f"Fetching available voices from {api_url}/voices...")
    status_code, data = make_http_request(f"{api_url}/voices")
    
    if status_code == 200 and data is not None:
        api_voices = []
        
        # Handle different response formats
        if isinstance(data, list):
            api_voices = data
        elif isinstance(data, dict) and "voices" in data:
            api_voices = data["voices"]
        elif isinstance(data, dict) and "status" in data and data.get("status") == "healthy":
            # Handle health endpoint format
            if "voices" in data:
                api_voices = data["voices"]
                
        # Combine with our custom voices and remove duplicates
        all_voices = list(set(api_voices + kokoro_voices))
        
        # If we got voices from the API, log success
        if api_voices:
            print(f"Retrieved {len(api_voices)} voices from API")
        
        return all_voices
    else:
        print(f"Couldn't get voices from API: {status_code}")
    
    # Return our default Kokoro voices if we can't get from API
    print("Using default Kokoro voices")
    return kokoro_voices

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
    
    print(f"Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"Using voice: {voice}, quality: {quality}, format: {format}")
    
    # Prepare the request payload
    payload = {
        "text": text,
        "voice": voice,
        "quality": quality,
        "format": format,
        "fiction": voice == "af_sky"  # Set fiction flag based on voice
    }
    
    print("Sending request to server...")
    
    # Make the API request
    status_code, response_data = make_http_request(
        url=f"{api_url}/tts",
        method="POST", 
        body=payload,
        timeout=60  # TTS can take a while for long texts
    )
    
    if status_code != 200:
        print(f"Error: API returned status code {status_code}")
        if isinstance(response_data, dict):
            print(f"Response: {json.dumps(response_data)}")
        return None
    
    # Parse the response data
    # For binary audio data (bytes)
    if isinstance(response_data, bytes):
        audio_data = response_data
        print(f"Successfully generated audio ({len(audio_data)/1024:.1f} KB)")
        
        # Save the audio if requested
        if save_path:
            save_audio(audio_data, save_path)
        
        # Play the audio if requested
        if play:
            play_audio(audio_data)
        
        return audio_data
        
    # For JSON responses (base64 encoded audio)
    elif isinstance(response_data, dict):
        try:
            # Try different known formats for base64 audio
            audio_base64 = None
            if "audio" in response_data and isinstance(response_data["audio"], str):
                audio_base64 = response_data["audio"]
            elif "audio_base64" in response_data:
                audio_base64 = response_data["audio_base64"]
                
            if audio_base64:
                audio_data = base64.b64decode(audio_base64)
                print(f"Successfully decoded base64 audio ({len(audio_data)/1024:.1f} KB)")
                
                if save_path:
                    save_audio(audio_data, save_path)
                
                if play:
                    play_audio(audio_data)
                
                return audio_data
        except Exception as e:
            print(f"Error decoding base64 audio: {str(e)}")
    
    print("Error: Unexpected response format")
    return None

def interactive_mode(api_url: str) -> None:
    """Run an interactive CLI session"""
    print(f"Kokoro TTS Interactive CLI")
    print(f"Connected to: {api_url}")

    # Get available voices
    voices = get_available_voices(api_url)
    print(f"Available voices: {', '.join(voices) or 'Unknown (using defaults)'}")

    while True:
        print("\n" + "="*50)
        print("New TTS Request")
        print("="*50)
        
        # Get user input
        text = input("Enter text to convert to speech (or 'q' to quit): ")
        if text.lower() == 'q':
            break
        
        # Select voice
        voice_options = voices if voices else ["en_US_1", "en_US_2", "en_GB_1"]
        if len(voice_options) > 1:
            print("Available voices:")
            for i, voice in enumerate(voice_options):
                print(f"  {i}: {voice}")
            try:
                voice_idx = int(input(f"Select voice (0-{len(voice_options)-1}): "))
                voice = voice_options[voice_idx]
            except (ValueError, IndexError):
                voice = voice_options[0]
                print(f"Invalid selection, using: {voice}")
        else:
            voice = voice_options[0]
        
        # Select quality
        quality_options = ["high", "medium", "low"]
        print("Quality options:")
        for i, q in enumerate(quality_options):
            print(f"  {i}: {q}")
        try:
            quality_idx = int(input("Select quality (0-2): "))
            quality = quality_options[quality_idx]
        except (ValueError, IndexError):
            quality = "medium"
            print(f"Invalid selection, using: {quality}")
        
        # Select format
        format_options = ["mp3", "wav"]
        print("Format options:")
        for i, f in enumerate(format_options):
            print(f"  {i}: {f}")
        try:
            format_idx = int(input("Select format (0-1): "))
            format = format_options[format_idx]
        except (ValueError, IndexError):
            format = "mp3"
            print(f"Invalid selection, using: {format}")
        
        # Choose whether to save
        save_choice = input("Save to file? (y/n): ").lower().strip()
        save_path = None
        if save_choice in ['y', 'yes']:
            save_path = input(f"Enter save path (default: output_{int(time.time())}.{format}): ")
            if not save_path:
                save_path = f"output_{int(time.time())}.{format}"
        
        # Choose whether to play
        play_choice = input("Play audio? (y/n): ").lower().strip()
        play = play_choice not in ['n', 'no']
        
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
            print("\nExiting...")
            sys.exit(0)

if __name__ == "__main__":
    main()
