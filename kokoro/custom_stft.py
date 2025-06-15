from attr import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomSTFT(nn.Module):
    """
    STFT/iSTFT without unfold/complex ops, using conv1d and conv_transpose1d.

    - forward STFT => Real-part conv1d + Imag-part conv1d
    - inverse STFT => Real-part conv_transpose1d + Imag-part conv_transpose1d + sum
    - avoids F.unfold, so easier to export to ONNX
    - uses replicate or constant padding for 'center=True' to approximate 'reflect' 
      (reflect is not supported for dynamic shapes in ONNX)
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",  # or 'constant'
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        # Number of frequency bins for real-valued STFT with onesided=True
        self.freq_bins = self.n_fft // 2 + 1

        # Build window and register as buffer
        window_tensor = self._prepare_window(win_length)
        self.register_buffer("window", window_tensor)

        # Prepare forward and backward weights
        self._prepare_forward_weights(window_tensor)
        self._prepare_backward_weights(window_tensor)
    
    def _prepare_window(self, win_length):
        """Prepare window tensor with proper padding/truncation"""
        assert win_length > 0, "Window length must be positive"
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        
        if self.win_length < self.n_fft:
            # Zero-pad up to n_fft
            extra = self.n_fft - self.win_length
            window_tensor = F.pad(window_tensor, (0, extra))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[: self.n_fft]
            
        return window_tensor
    
    def _register_weight_buffer(self, name, data, unsqueeze=True):
        """Convert numpy array to torch tensor and register as buffer"""
        tensor = torch.from_numpy(data).float()
        if unsqueeze:
            tensor = tensor.unsqueeze(1)
        self.register_buffer(name, tensor)
        
    def _create_angle_matrix(self, n_range, k_range, transpose=False):
        """Create angle matrix for DFT calculation"""
        if transpose:
            angle = 2 * np.pi * np.outer(n_range, k_range) / self.n_fft
            return angle.T  # Shape: (freq_bins, n_fft)
        else:
            return 2 * np.pi * np.outer(k_range, n_range) / self.n_fft  # Shape: (freq_bins, n_fft)
    
    def _prepare_forward_weights(self, window_tensor):
        """Prepare forward STFT weights"""
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        
        # Calculate angle matrix
        angle = self._create_angle_matrix(n, k)
        
        # Calculate real and imaginary components
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)  # Note negative sign for PyTorch STFT compatibility
        
        # Apply window to both components
        forward_window = window_tensor.numpy()
        forward_real = dft_real * forward_window
        forward_imag = dft_imag * forward_window
        
        # Register as buffers
        self._register_weight_buffer("weight_forward_real", forward_real)
        self._register_weight_buffer("weight_forward_imag", forward_imag)
    
    def _prepare_backward_weights(self, window_tensor):
        """Prepare inverse STFT weights"""
        inv_scale = 1.0 / self.n_fft
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        
        # Calculate transposed angle matrix
        angle_t = self._create_angle_matrix(n, k, transpose=True)
        
        # Calculate inverse DFT components
        idft_cos = np.cos(angle_t)
        idft_sin = np.sin(angle_t)
        
        # Apply scaled window
        inv_window = window_tensor.numpy() * inv_scale
        backward_real = idft_cos * inv_window
        backward_imag = idft_sin * inv_window
        
        # Register as buffers
        self._register_weight_buffer("weight_backward_real", backward_real)
        self._register_weight_buffer("weight_backward_imag", backward_imag)
        


    def transform(self, waveform: torch.Tensor):
        """
        Forward STFT => returns magnitude, phase
        Output shape => (batch, freq_bins, frames)
        """
        # waveform shape => (B, T).  conv1d expects (B, 1, T).
        # Optional center pad
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)

        x = waveform.unsqueeze(1)  # => (B, 1, T)
        # Convolution to get real part => shape (B, freq_bins, frames)
        real_out = F.conv1d(
            x,
            self.weight_forward_real,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # Imag part
        imag_out = F.conv1d(
            x,
            self.weight_forward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )

        # magnitude, phase
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        # Handle the case where imag_out is 0 and real_out is negative to correct ONNX atan2 to match PyTorch
        # In this case, PyTorch returns pi, ONNX returns -pi
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase


    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        """
        Inverse STFT => returns waveform shape (B, T).
        """
        # magnitude, phase => (B, freq_bins, frames)
        # Re-create real/imag => shape (B, freq_bins, frames)
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)

        # conv_transpose wants shape (B, freq_bins, frames). We'll treat "frames" as time dimension
        # so we do (B, freq_bins, frames) => (B, freq_bins, frames)
        # But PyTorch conv_transpose1d expects (B, in_channels, input_length)
        real_part = real_part  # (B, freq_bins, frames)
        imag_part = imag_part

        # real iSTFT => convolve with "backward_real", "backward_imag", and sum
        # We'll do 2 conv_transpose calls, each giving (B, 1, time),
        # then add them => (B, 1, time).
        real_rec = F.conv_transpose1d(
            real_part,
            self.weight_backward_real,  # shape (freq_bins, 1, filter_length)
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        imag_rec = F.conv_transpose1d(
            imag_part,
            self.weight_backward_imag,
            bias=None,
            stride=self.hop_length,
            padding=0,
        )
        # sum => (B, 1, time)
        waveform = real_rec - imag_rec  # typical real iFFT has minus for imaginary part

        # If we used "center=True" in forward, we should remove pad
        if self.center:
            pad_len = self.n_fft // 2
            # Because of transposed convolution, total length might have extra samples
            # We remove `pad_len` from start & end if possible
            waveform = waveform[..., pad_len:-pad_len]

        # If a specific length is desired, clamp
        if length is not None:
            waveform = waveform[..., :length]

        # shape => (B, T)
        return waveform

    def forward(self, x: torch.Tensor):
        """
        Full STFT -> iSTFT pass: returns time-domain reconstruction.
        Same interface as your original code.
        """
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
