from pathlib import Path

import numpy as np
import scipy.fft
from scipy.io import wavfile


class SweptSine:

    def __init__(self, fs, f1, f2, duration):

        self.fs = fs
        self.f1 = f1
        self.f2 = f2
        self.target_duration = duration  # save the target duration

        self._L = self._calculate_L(self.f1, self.f2, self.target_duration)
        self._T = self._calculate_T(self.f1, self.f2, self._L)
        self.actal_duration = self._T  # alias for the actual calculated duration
        self._t = self._generate_t(self._T, self.fs)

        self.sweep = self._generate_sweep(self.f1, self._L, self._t)
        self.inverse = self._generate_inverse(self._t, self._L, self.sweep)
        self.impulse_response = np.zeros(self.N)

        # Precompute parts of the deconvolution
        self._X_inverse = scipy.fft.rfft(self.inverse, n=self.N)
        self._reference_impulse_response = self._deconvolution(self.sweep)

    @classmethod
    def init_from_sweep_wav(cls, filepath):
        """Create an instance from the init sweep parameters found in the filepath."""
        fs, f1, f2, duration = cls._parameters_from_wav_filepath(filepath)
        return cls(fs, f1, f2, duration)

    #### Utils

    def to_dBFS(self, data, floor_dB=-120):
        """Transform a signal and scale to dBFS."""
        y = np.abs(scipy.fft.rfft(data, n=len(data)))
        y /= len(data) / 2  # normalise
        y = np.clip(y, 10 ** (floor_dB / 20), None)
        y_dB = 20 * np.log10(y)

        x = np.linspace(0, self.fs / 2, num=len(y_dB))

        return x, y_dB

    #### WAV Files

    def _construct_wav_filepath(self, path=".", prefix="sweep"):
        """Create a filepath with the sweep parameters appended to the filename."""
        parameters = f"{self.fs}-{self.f1}-{self.f2}-{self.target_duration}"
        filename = f"{prefix}-{parameters}.wav"
        filepath = Path(path) / filename
        return filepath

    @staticmethod
    def _parameters_from_wav_filepath(filepath):
        """Extract the sweep parameters from a filepath created with `_construct_wav_filepath()`."""
        filename = Path(filepath).stem
        fs, f1, f2, duration = map(float, filename.split("-")[-4:])
        return fs, f1, f2, duration

    def save_sweep_as_wav(self, path=".", prefix="sweep"):
        """Save the sweep signal to wav file with the parameters appended to the filename."""
        filepath = self._construct_wav_filepath(path, prefix)
        wavfile.write(filepath, self.fs, self.sweep)
        return filepath

    def deconvolve_from_wav(self, filepath):
        filepath = Path(filepath)
        fs, measurement = wavfile.read(filepath)
        if fs != self.fs:
            raise ValueError(
                f"Measurement {filepath} sample rate ({fs}) does not match the sweep sample rate ({self.fs})."
            )
        return self.deconvolve(measurement)

    #### Sweep Parameters

    @staticmethod
    def _calculate_L(f1, f2, target_T):
        """Calculate the rate of frequency increase, L, required for the target sweep duration, T (Novak et al. 2015, eq. 49)."""
        L = (1 / f1) * np.round((f1 / np.log(f2 / f1)) * target_T)
        return L

    @staticmethod
    def _calculate_T(f1, f2, L):
        """Calculate the actual duration, T, required for phase synchronicity (Novak et al. 2015, eq. 48)."""
        actual_T = L * np.log(f2 / f1)
        return actual_T

    @staticmethod
    def _generate_t(T, fs):
        """Generate a vector of samples in time."""
        T_samples = int(np.ceil(T * fs))
        t = np.arange(T_samples) / fs
        return t

    @staticmethod
    def _generate_sweep(f1, L, t):
        """Generate the synchronised swept sine (Novak et al. 2015, eq. 47)."""
        sweep = np.sin(2 * np.pi * f1 * L * np.exp(t / L))
        return sweep

    @staticmethod
    def _generate_inverse(t, L, sweep):
        """Generate the inverse filter using Farina's amplitude-modulated, time-reversed method (2000)."""
        env = np.exp(t * (1 / L))
        inverse = np.flip(np.copy(sweep), axis=0) / env
        return inverse

    #### Analysis

    @property
    def N(self):
        return len(self.sweep) + len(self.inverse) - 1

    def _deconvolution(self, measurement):
        """Perform the matched linear deconvolution using the precomputed inverse filter (X~)."""
        N = self.N
        Y = scipy.fft.rfft(measurement, n=N)
        impulse_response = scipy.fft.irfft(Y * self._X_inverse, n=N)
        return impulse_response

    def deconvolve(self, measurement, normalise=True):
        """Deconvolve the measurement with the inverse filter and optionally normalise the output impulse response."""

        self.impulse_response = self._deconvolution(measurement)

        if normalise:
            self.impulse_response /= np.max(np.abs(self._reference_impulse_response))

        return self.impulse_response

    def nth_harmonic_time_delay(self, n):
        """Calculate the time delay in seconds for the nth harmonic (Novak et al. 2015, eq. 34)."""
        return self._L * np.log(n)

    def nth_harmonic_sample_delay(self, n):
        """Calculate the time delay in samples at fs for the nth harmonic."""
        delta_t = self.nth_harmonic_time_delay(n)
        return int(np.round(delta_t * self.fs))

    def sweep_frequency_at_time(self, t):
        """Calculate the frequency of the sweep at a given time (Novak et al. 2015, eq. 35)."""
        return self.f1 * np.exp(t / self._L)

    def get_fundamental_impulse_response(self):
        """Get only the portion of the impulse response after t=0 (single sided)."""
        return self.impulse_response[len(self.impulse_response) // 2 :]

    def get_harmonic_impulse_response(self, n):
        """Get the (centred) impulse response of the nth harmonic."""
        harmonic_idx = self.N // 2 - self.nth_harmonic_sample_delay(n)
        next_harmonic_idx = self.N // 2 - self.nth_harmonic_sample_delay(n + 1)

        max_half_window = (harmonic_idx - next_harmonic_idx) // 2
        start_idx = harmonic_idx - max_half_window
        end_idx = harmonic_idx + max_half_window

        return self.impulse_response[start_idx:end_idx]
