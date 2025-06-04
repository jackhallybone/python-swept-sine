import numpy as np
from numpy.testing import assert_array_equal

from swept_sine import SweptSine


def test_init_creates_signals():
    """Verify that the class init creates the sweep and inverse signals."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    assert isinstance(swept_sine.sweep, np.ndarray)
    assert isinstance(swept_sine.inverse, np.ndarray)
    assert isinstance(swept_sine.impulse_response, np.ndarray)


def test_init_calculations_are_deterministic():
    """Verify that the init arguments and calculations are reproducible."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    assert_array_equal(swept_sine.sweep, SweptSine(20000, 100, 1000, 4).sweep)


def test_init_from_sweep_wav(tmp_path):
    """Verify that an instance can be recreated from the parameters in a filename."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="init_test")
    new_swept_sine = SweptSine.init_from_sweep_wav(filepath)
    assert_array_equal(new_swept_sine.sweep, swept_sine.sweep)  # sweeps match


#### WAV Files


def test__construct_wav_filepath():
    """Verify that a wav filename can be constructed including the sweep parameters."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine._construct_wav_filepath(path="test/path", prefix="my_sweep")
    from pathlib import Path

    assert isinstance(filepath, Path)
    assert filepath == Path(f"test/path/my_sweep-20000-100-1000-4.wav")


def test__parameters_from_wav_filepath():
    """Verify that the sweep parameters can be read from a properly formatted filename."""
    filepath = "test/path/my_sweep-20000-100-1000-4.wav"
    assert SweptSine._parameters_from_wav_filepath(filepath) == (20000, 100, 1000, 4)


def test_save_sweep_as_wav(tmp_path):
    """Verify that saving to wav maintains the sweep signal format, etc."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="save_test")
    from scipy.io import wavfile

    fs, data = wavfile.read(filepath)
    assert fs == swept_sine.fs
    assert_array_equal(data, swept_sine.sweep)  # sweeps match


#### Sweep Parameters

# def test__calculate_L():

# def test__calculate_T():

# def test__generate_t():

# def test__generate_sweep():

# def test__generate_inverse():
# As per Farina 2000
# 20*log10(_inverse_envelope(t, L))[-1] == -6*np.log2(f2/f1)

#### Analysis


def test_N():
    swept_sine = SweptSine(20000, 10, 500, 1)
    assert len(swept_sine.sweep) == len(swept_sine.inverse) == len(swept_sine._t)
    assert swept_sine.N == len(swept_sine._t) + len(swept_sine._t) - 1
