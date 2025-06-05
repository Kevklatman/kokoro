import torch
import numpy as np

def add_breathiness(audio, amount=0.0):
    """Add breathiness by mixing in filtered noise"""
    if amount <= 0:
        return audio
    
    # Generate noise matching the audio shape
    noise = torch.randn_like(audio) * 0.1
    
    # Apply soft lowpass filter to the noise to make it sound more like breath
    noise_filt = torch.nn.functional.conv1d(
        noise.view(1, 1, -1),
        torch.ones(1, 1, 32).to(noise.device) / 32,
        padding='same'
    ).view(-1)
    
    # Mix with original
    return audio * (1 - amount * 0.5) + noise_filt * amount

def add_tenseness(audio, amount=0.0):
    """Add tenseness by applying soft clipping/compression"""
    if amount <= 0:
        return audio
        
    # Simple soft clipping to create the impression of tension
    gain = 1.0 + amount * 1.5
    audio_amplified = audio * gain
    
    # Soft clipping function
    return torch.tanh(audio_amplified) / gain

def add_jitter(audio, amount=0.0):
    """Add jitter by slightly deforming time and amplitude"""
    if amount <= 0:
        return audio
    
    # Time jitter - small random phase variations
    audio_length = audio.shape[0]
    
    # Create time indices with small random shifts
    indices = torch.arange(audio_length).float()
    jitter_amount = amount * 2.0  # Scale factor
    
    # Generate smooth random variations
    variations = torch.sin(indices / 20) * torch.sin(indices / 73) * torch.sin(indices / 127)
    variations = variations * jitter_amount
    
    # Create new indices with jitter
    new_indices = indices + variations
    
    # Ensure indices are in bounds
    new_indices = torch.clamp(new_indices, 0, audio_length - 1)
    
    # Convert to integer indices for interpolation
    idx_floor = new_indices.floor().long()
    idx_ceil = new_indices.ceil().long()
    
    # Interpolation weights
    alpha = new_indices - idx_floor
    
    # Interpolate (simple linear interpolation)
    result = audio[idx_floor] * (1 - alpha) + audio[idx_ceil] * alpha
    
    return result

def add_sultry(audio, amount=0.0):
    """Add sultry effect with lowered pitch and smoother transitions"""
    if amount <= 0:
        return audio
    
    # Convert to torch tensor if needed
    is_numpy = isinstance(audio, np.ndarray)
    if is_numpy:
        audio = torch.from_numpy(audio).float()
    
    # Create a safer version of the smoothing operation
    try:
        # Apply smoothing to transitions
        audio_len = audio.shape[0]
        window_size = min(128, max(16, audio_len // 8))  # Adapt window size to audio length
        
        # Make sure we're working with a valid tensor for conv1d
        audio_3d = audio.view(1, 1, -1)  
        
        # Apply the smoothing filter
        smoothed = torch.nn.functional.avg_pool1d(
            audio_3d,
            kernel_size=window_size,
            stride=1,
            padding=window_size//2
        )
        
        # Ensure same shape as original
        smoothed = smoothed.view(-1)[:audio_len]
        
        # If shapes still don't match, just use a simpler approach
        if smoothed.shape[0] != audio.shape[0]:
            # Fallback to a simpler smoothing approach
            smoothed = audio.clone()
            for i in range(1, audio_len-1):
                smoothed[i] = (audio[i-1] + audio[i] + audio[i+1]) / 3.0
        
        # Mix original with smoothed version
        mix_ratio = amount * 0.6
        result = audio * (1 - mix_ratio) + smoothed * mix_ratio
        
        # Apply gentle time stretching effect only if audio is long enough
        if audio_len > 1000:
            try:
                # Slight time stretch to slow down delivery
                stretch_factor = 1.0 + (amount * 0.1)  # Max 10% slower
                orig_len = result.shape[0]
                new_len = int(orig_len * stretch_factor)
                
                # Create interpolation indices
                indices = torch.linspace(0, orig_len - 1, new_len)
                idx_floor = indices.floor().long()
                idx_ceil = indices.ceil().long().clamp(max=orig_len-1)
                alpha = indices - idx_floor
                
                # Interpolate to create the stretched audio
                stretched = result[idx_floor] * (1 - alpha) + result[idx_ceil] * alpha
                result = stretched[:orig_len]  # Keep original length
            except Exception:
                # If stretching fails, just continue with unstretched audio
                pass
                
    except Exception as e:
        # If anything fails, just return the original audio with minimal processing
        result = audio
    
    # Convert back if needed
    if is_numpy:
        result = result.numpy()
        
    return result

def apply_emotion_effects(audio, breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    """Apply all emotion effects to audio"""
    # Convert to torch tensor if it's numpy
    is_numpy = isinstance(audio, np.ndarray)
    if is_numpy:
        audio = torch.from_numpy(audio).float()
        
    # Apply effects in sequence
    audio = add_breathiness(audio, breathiness)
    audio = add_tenseness(audio, tenseness) 
    audio = add_jitter(audio, jitter)
    audio = add_sultry(audio, sultry)
    
    # Convert back if needed
    if is_numpy:
        audio = audio.numpy()
        
    return audio
