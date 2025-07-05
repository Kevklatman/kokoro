#!/usr/bin/env python3
"""
Test script to check all available voices in the cloud deployment
"""
import requests
import json
import time

# Cloud Run URL
BASE_URL = "https://kokoro-tts-696551753574.us-central1.run.app"

def test_voice(voice_name):
    """Test if a voice is available"""
    url = f"{BASE_URL}/tts/"
    data = {
        "text": "Test",
        "voice": voice_name,
        "format": "wav",
        "quality": "high"
    }
    
    try:
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            return True, response.headers.get('content-length', 'unknown')
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    # List of voices to test (based on the voice files we saw in logs)
    voices_to_test = [
        'af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 
        'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 
        'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 
        'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 
        'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 
        'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 
        'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 
        'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 
        'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 
        'zm_yunxia', 'zm_yunyang'
    ]
    
    print("Testing voices in cloud deployment...")
    print("=" * 50)
    
    working_voices = []
    failed_voices = []
    
    for i, voice in enumerate(voices_to_test, 1):
        print(f"[{i}/{len(voices_to_test)}] Testing {voice}...", end=" ")
        success, result = test_voice(voice)
        
        if success:
            print(f"✅ Working ({result} bytes)")
            working_voices.append(voice)
        else:
            print(f"❌ Failed: {result}")
            failed_voices.append((voice, result))
        
        # Small delay to avoid overwhelming the service
        time.sleep(0.5)
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {len(working_voices)} working voices, {len(failed_voices)} failed")
    print("\nWorking voices:")
    for voice in working_voices:
        print(f"  ✅ {voice}")
    
    if failed_voices:
        print("\nFailed voices:")
        for voice, error in failed_voices:
            print(f"  ❌ {voice}: {error}")
    
    # Check health endpoint
    print("\n" + "=" * 50)
    print("Health endpoint status:")
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Status: {health_data.get('status')}")
            print(f"Voices reported: {health_data.get('voices', [])}")
        else:
            print(f"Health check failed: HTTP {health_response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    main() 