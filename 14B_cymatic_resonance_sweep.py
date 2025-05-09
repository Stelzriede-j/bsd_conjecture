
# Cymatic Resonance Sweep Script with Structure Detection
# Author: Jacob Stelzriede (with OpenAI)
# Date: April 2025

"""
This script sweeps through a specified frequency range and detects symbolic lock-in events
based on resonance structure emergence, using a scalar wave model over a radial field.

Key Features:
- Simulates harmonic resonance on a 2D circular grid
- Applies FFT energy detection to measure symbolic emergence
- Logs frequencies where structural lock occurs
- Calculates delta-f (gap between successive lock-ins)
- Designed to scale across coarse or fine frequency bands (e.g. 1Hz to 10kHz, 10kHz to 100kHz)

Use Cases:
- Cymatic-based resonance field modeling
- Visualizing quantized structure emergence (mass gap mapping)
- Symbolic resonance ladder tracking
- BSD-inspired field emergence studies

Parameters:
- freq_start: beginning of frequency range (Hz)
- freq_stop: end of frequency range (Hz)
- num_steps: number of frequency steps between start and stop
- structure_threshold: structure score required to count as lock-in
- smoothing: Gaussian blur to apply to field before FFT detection
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, fftshift

def cymatic_resonance_sweep(
    freq_start=1,
    freq_stop=10000,
    num_steps=2000,
    structure_threshold=0.055,
    smoothing=1.2,
    Nx=100,
    Ny=100
):
    # Grid setup
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    frequencies = np.linspace(freq_start, freq_stop, num_steps)
    lock_frequencies = []
    symbolic_scores = []
    delta_f = []

    previous_lock = None

    def detect_structure(pattern):
        spectrum = np.abs(fftshift(fft2(pattern)))
        center = spectrum[Nx//2 - 10:Nx//2 + 10, Ny//2 - 10:Ny//2 + 10]
        energy_ratio = np.sum(center) / np.sum(spectrum)
        return energy_ratio

    for f in frequencies:
        pattern = np.sin(2 * np.pi * f * R)
        smoothed = gaussian_filter(pattern, sigma=smoothing)
        score = detect_structure(smoothed)

        if score > structure_threshold:
            lock_frequencies.append(float(f))
            symbolic_scores.append(float(score))
            if previous_lock is not None:
                delta_f.append(float(f - previous_lock))
            previous_lock = f

    return lock_frequencies, symbolic_scores, delta_f

# Example usage
if __name__ == "__main__":
    locks, scores, gaps = cymatic_resonance_sweep(freq_start=1, freq_stop=10000, num_steps=2000)
    print("First 10 lock-in frequencies:", [round(v, 2) for v in locks[:10]])
    print("First 10 symbolic scores:", [round(v, 3) for v in scores[:10]])
    print("First 10 delta-f gaps:", [round(v, 3) for v in gaps[:10]])
