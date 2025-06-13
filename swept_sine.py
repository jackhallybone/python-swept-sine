from pathlib import Path

import numpy as np
import scipy.fft
from scipy.io import wavfile


class SweptSine:

    re_init_parameters = {  # The parameters/arguments required to re-create a sweep.
        "fs": int,
        "f1": float,
        "f2": float,
        "target_duration": float,
        "sweep_dBFS": float,
        "fade_in": float,
        "fade_out": float,
        "fade_shape": str,
        "pad_start": float,
        "pad_end": float,
    }

    def __init__(
        self,
        fs,
        f1,
        f2,
        duration,
        sweep_dBFS=0,
        fade_in=0,
        fade_out=0,
        fade_shape="cosine",
        pad_start=0,
        pad_end=0,
    ):

        if f2 <= f1:
            # https://www.ap.com/blog/why-are-chirps-always-swept-from-low-to-high
            raise ValueError("Sweep must be high to low (f2>f1)")

        self.fs = fs
        self.f1 = f1
        self.f2 = f2
        self.target_duration = duration  # save the target duration
        self.sweep_dBFS = sweep_dBFS
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.fade_shape = fade_shape
        self.pad_start = pad_start
        self.pad_end = pad_end

        # Calculate the sweep from the user provided parameters
        self._L = self._calculate_L(self.f1, self.f2, self.target_duration)
        self._T = self._calculate_T(self.f1, self.f2, self._L)
        self.actual_duration = self._T  # alias for the actual calculated duration

        # Create a time series of shape (samples, channels) for creating the signals
        self._t = self._generate_t(self._T, self.fs)
        self._t = self._enforce_2d_row_major(self._t)

        # self._t = np.repeat(self._t, 10, axis=1) # example to make a 10 channel system

        # Generate the sweep, optionally with fades, and the inverse filter signals
        self.sweep = self._generate_sweep(self.f1, self._L, self._t)
        if self.fade_in > 0 or self.fade_out > 0:
            self.sweep = self._fade_in_out(
                self.sweep, self.fade_in, self.fade_out, self.fade_shape
            )
        self.inverse = self._generate_inverse(self._t, self._L, self.sweep)

        # Apply and zero padding to the sweep and apply it to the inverse backwards
        if self.pad_start > 0 or self.pad_end > 0:
            self.sweep = self._zero_pad_start_end(
                self.sweep, self.pad_start, self.pad_end
            )
            self.inverse = self._zero_pad_start_end(
                self.inverse, self.pad_end, self.pad_start
            )

        # Precompute components of the deconvolution to save time later
        self._X_inverse = scipy.fft.rfft(self.inverse, n=self.N, axis=0)
        self._reference_impulse_response = self._deconvolution(self.sweep)

        # Scale sweep to dBFS
        if self.sweep_dBFS != 0:
            self.sweep = self._scale_by_dBFS(self.sweep, self.sweep_dBFS)

    @classmethod
    def init_from_sweep_filename(cls, filepath):
        """Create an instance from the sweep parameters found in the filename."""
        params = cls._parameters_from_wav_filename(filepath)
        return cls(*params)

    #### Utils

    @staticmethod
    def _enforce_2d_row_major(data):
        """Enforce a signal shape of (samples, channels), including for mono."""
        data = np.asarray(data)
        if data.ndim == 1:  # Expand 1D mono to 2D mono (samples, 1)
            return data[:, np.newaxis]
        elif data.ndim == 2:
            rows, columns = data.shape
            if rows < columns:  # likely (channels, samples) so transpose
                return data.transpose()
            return data
        else:
            raise ValueError("Audio data must be be 1D or 2D.")

    @staticmethod
    def _scale_by_dBFS(data, dBFS):
        return data * np.power(10, dBFS / 20)

    @staticmethod
    def _seconds_to_samples(t, fs):
        """Convert a duration in seconds to a number of samples at a sampling rate."""
        return round(t * fs)

    @staticmethod
    def _samples_to_seconds(length, fs):
        """Convert a number of samples at a sampling rate to a duration in seconds."""
        return length / fs

    def _fade_in_out(self, data, fade_in, fade_out, fade_shape="cosine"):
        """Fade a signal in from 0 and/or out to 0."""
        data = self._enforce_2d_row_major(data)

        cls = type(self)
        fade_in_length = cls._seconds_to_samples(fade_in, self.fs)
        fade_out_length = cls._seconds_to_samples(fade_out, self.fs)

        if fade_in_length + fade_out_length > data.shape[0]:
            raise ValueError(
                f"Fade in + fade out ({fade_in + fade_out}s) exceeds signal length "
                f"({cls._samples_to_seconds(data.shape[0], self.fs)}s)."
            )

        envelope = np.ones((data.shape[0], 1))

        if fade_in_length > 0:
            envelope[:fade_in_length, :] = cls._create_fade_envelope(
                fade_in_length, 0, 1, fade_shape
            )
        if fade_out_length > 0:
            envelope[-fade_out_length:, :] = cls._create_fade_envelope(
                fade_out_length, 1, 0, fade_shape
            )

        faded_data = data * envelope

        return faded_data

    @staticmethod
    def _create_fade_envelope(length, start, end, fade_shape):
        """Draw a shaped envelope for fading a signal."""
        if fade_shape == "linear":
            curve = np.linspace(start, end, length, endpoint=True)
        elif fade_shape == "cosine":
            x = np.linspace(0, 1, length)
            curve = 0.5 * (1 - np.cos(np.pi * x))
            curve = curve * (end - start) + start
        else:
            raise ValueError(
                f"Fade shape must be 'linear' or 'cosine' not '{fade_shape}'."
            )
        return curve[:, np.newaxis]

    def _zero_pad_start_end(self, data, pad_start, pad_end):
        """Zero pad the start and end of a signal."""
        cls = type(self)
        pad_start_length = cls._seconds_to_samples(pad_start, self.fs)
        pad_end_length = cls._seconds_to_samples(pad_end, self.fs)

        padded_data = np.pad(
            data,
            pad_width=((pad_start_length, pad_end_length), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return padded_data

    #### WAV Files

    def _construct_wav_filepath(self, path=".", prefix="sweep"):
        """Create a filepath with the sweep parameters appended to the filename."""
        param_suffix = "_".join(str(getattr(self, p)) for p in self.re_init_parameters)
        filename = f"{prefix}-params_{param_suffix}.wav"
        filepath = Path(path) / filename
        return filepath

    @classmethod
    def _parameters_from_wav_filename(cls, filepath):
        """Extract the sweep parameters from a filename created with `_construct_wav_filepath()`."""
        filename = Path(filepath).stem
        parts = filename.split("_")
        params = parts[-len(cls.re_init_parameters) :]
        cast_params = [
            to_type(param)
            for to_type, param in zip(cls.re_init_parameters.values(), params)
        ]
        return cast_params

    def save_sweep_as_wav(self, path=".", prefix="sweep"):
        """Save the sweep signal to a wav file with the parameters appended to the filename."""
        filepath = self._construct_wav_filepath(path, prefix)
        wavfile.write(filepath, self.fs, self.sweep)
        return filepath

    def _read_from_wav(self, filepath):
        """Read an audio file from a wav file and enforce 2D (samples, channels) shape."""
        filepath = Path(filepath)
        fs, measurement = wavfile.read(filepath)
        if fs != self.fs:
            raise ValueError(
                f"Measurement {filepath} sample rate ({fs}) does not match the sweep sample rate ({self.fs})."
            )
        measurement = self._enforce_2d_row_major(
            measurement
        )  # wavfile.read returns mono as 1D
        return measurement

    def deconvolve_from_wav(self, filepath):
        """Read measurement data from a wav file and deconvolve it"""
        measurement = self._read_from_wav(filepath)
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
        envelope = np.exp(t * (1 / L))
        inverse = np.flip(np.copy(sweep), axis=0) / envelope
        return inverse

    #### Sweep Analysis

    def sweep_frequency_at_time(self, t):
        """Calculate the frequency of the sweep at a given time (Novak et al. 2015, eq. 35)."""
        return self.f1 * np.exp(t / self._L)

    #### Analysis

    @property
    def N(self):
        return len(self.sweep) + len(self.inverse) - 1

    def _deconvolution(self, measurement):
        """Perform the matched linear deconvolution using the precomputed inverse filter (X~)."""
        N = self.N
        Y = scipy.fft.rfft(measurement, n=N, axis=0)
        impulse_response = scipy.fft.irfft(Y * self._X_inverse, n=N, axis=0)
        return impulse_response

    def deconvolve(self, measurement, normalise=True):
        """Deconvolve the measurement with the inverse filter and optionally normalise the output impulse response."""

        measurement = self._enforce_2d_row_major(measurement)

        impulse_response = self._deconvolution(measurement)

        if normalise:
            impulse_response /= np.max(
                np.abs(self._reference_impulse_response)
            )

        return impulse_response

    def impulse_response_frequency_response(self, impulse_response, floor_dB=-120):
        frequency_response = np.abs(
            scipy.fft.rfft(impulse_response, n=len(impulse_response))
        )

        frequency_response = np.clip(frequency_response, 10 ** (floor_dB / 20), None)
        frequency_response_dB = 20 * np.log10(frequency_response)

        bin_frequencies = np.linspace(0, self.fs / 2, num=len(frequency_response_dB))

        return bin_frequencies, frequency_response_dB

    def nth_harmonic_time_delay(self, n):
        """Calculate the time delay in seconds for the nth harmonic (Novak et al. 2015, eq. 34)."""
        return self._L * np.log(n)

    def nth_harmonic_sample_delay(self, n):
        """Calculate the time delay in samples at fs for the nth harmonic."""
        delta_t = self.nth_harmonic_time_delay(n)
        return type(self)._seconds_to_samples(delta_t, self.fs)

    @staticmethod
    def get_fundamental_impulse_response(impulse_response):
        """Get only the portion of the impulse response after t=0 (single sided)."""
        return impulse_response[len(impulse_response) // 2 :]

    def get_harmonic_impulse_response(self, impulse_response, n):
        """Get the (centred) impulse response of the nth harmonic."""
        harmonic_idx = self.N // 2 - self.nth_harmonic_sample_delay(n)
        next_harmonic_idx = self.N // 2 - self.nth_harmonic_sample_delay(n + 1)

        max_half_window = (harmonic_idx - next_harmonic_idx) // 2
        start_idx = harmonic_idx - max_half_window
        end_idx = harmonic_idx + max_half_window

        return impulse_response[start_idx:end_idx]

    #### Convolve

    @classmethod
    def convolve(cls, data, impulse_response, N=None, next_fast_len=False):
        """Convolve a signal with an impulse response."""
        data = cls._enforce_2d_row_major(data)
        impulse_response = cls._enforce_2d_row_major(impulse_response)

        if N is None:
            N = data.shape[0] + impulse_response.shape[0] - 1

        if next_fast_len:
            N = scipy.fft.next_fast_len(N, real=True)

        X = scipy.fft.rfft(data, n=N, axis=0)
        H = scipy.fft.rfft(impulse_response, n=N, axis=0)
        Y = X * H
        y = scipy.fft.irfft(Y, n=N, axis=0)

        return y
