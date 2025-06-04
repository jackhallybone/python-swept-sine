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

    @classmethod
    def init_from_sweep_wav(cls, filepath):
        """Create an instance from the init sweep parameters found in the filepath."""
        fs, f1, f2, duration = cls._parameters_from_wav_filepath(filepath)
        return cls(fs, f1, f2, duration)

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

    def deconvolve(self, measurement):

        N = self.N
        IR = scipy.fft.irfft(
            scipy.fft.rfft(measurement, n=N, axis=0)
            * scipy.fft.rfft(self.inverse, n=N, axis=0),
            n=N,
            axis=0,
        )

        IR = IR / np.float32(self.fs**2)  # normalise for sample rate
        IR = IR / self._L * self.f2  # normalise for frequency range
        IR = (
            IR * np.float32(N) * 2
        )  # normalise such that the raw signals produce a 0dBFS output
        # IR = IR * (np.power(10, -scale_to_dBFS/20)) # Optionally, normalise out the stimulus level

        return IR
