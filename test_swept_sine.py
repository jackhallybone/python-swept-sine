import numpy as np
import pytest
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


#### Utils


# def test_to_dBFS():


#### WAV Files


def test__construct_wav_filepath():
    """Verify that a wav filename can be constructed including the sweep parameters."""
    from pathlib import Path

    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine._construct_wav_filepath(path="test/path", prefix="my_sweep")
    assert isinstance(filepath, Path)
    assert filepath == Path(f"test/path/my_sweep-20000-100-1000-4.wav")


def test__parameters_from_wav_filepath():
    """Verify that the sweep parameters can be read from a properly formatted filename."""
    filepath = "test/path/my_sweep-20000-100-1000-4.wav"
    assert SweptSine._parameters_from_wav_filepath(filepath) == (20000, 100, 1000, 4)


def test_save_sweep_as_wav(tmp_path):
    """Verify that saving to wav maintains the sweep signal format, etc."""
    from scipy.io import wavfile

    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="save_test")
    fs, data = wavfile.read(filepath)
    assert fs == swept_sine.fs
    assert_array_equal(data, swept_sine.sweep)  # sweeps match


def test_deconvolve_from_wav(tmp_path):
    """Verify that measurements can be deconvolved directly from a wav file."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="deconvolve_test")
    wav_impulse_response = swept_sine.deconvolve_from_wav(filepath)
    direct_impulse_response = swept_sine.deconvolve(swept_sine.sweep)
    assert_array_equal(wav_impulse_response, direct_impulse_response)


#### Sweep Parameters

# def test__calculate_L():

# def test__calculate_T():

# def test__generate_t():

# def test__generate_sweep():

# def test__generate_inverse():
# As per Farina 2000
# 20*log10(_inverse_envelope(t, L))[-1] == -6*np.log2(f2/f1)


def test_sweep_start_end_phase_synchronicity():
    """Verify that the start and end of the sweep are (close to) 0."""
    swept_sine = SweptSine(48000, 10, 500, 1)
    assert swept_sine.sweep[0] == pytest.approx(0)
    assert swept_sine.sweep[-1] == pytest.approx(0, abs=0.01)


def test_sweep_harmonic_phase_synchronicity():
    """Verify that each harmonic of f1 in the sweep 'starts' at or close to a positive-going zero cross."""
    f1 = 10
    swept_sine = SweptSine(48000, f1, 500, 1)
    for n in range(2, 10, 1):
        idx = swept_sine.nth_harmonic_sample_delay(n)
        assert (
            swept_sine.sweep[idx - 1]
            < swept_sine.sweep[idx]
            < swept_sine.sweep[idx + 1]
        )  # postive-going
        assert (
            swept_sine.sweep[idx - 1] < 0 and swept_sine.sweep[idx + 1] > 0
        )  # zero-cross


#### Analysis


def test_N():
    swept_sine = SweptSine(20000, 10, 500, 1)
    assert len(swept_sine.sweep) == len(swept_sine.inverse) == len(swept_sine._t)
    assert swept_sine.N == len(swept_sine._t) + len(swept_sine._t) - 1


def test_deconvolve():
    """Validate that the deconvolution and normalisation seems roughly correct."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    swept_sine.deconvolve(swept_sine.sweep)
    assert np.max(np.abs(swept_sine.impulse_response)) == pytest.approx(1)
    other_swept_sine = SweptSine(5000, 10, 500, 15)
    other_swept_sine.deconvolve(other_swept_sine.sweep)
    assert np.max(np.abs(other_swept_sine.impulse_response)) == pytest.approx(1)
    other_swept_sine.deconvolve(other_swept_sine.sweep, normalise=False)
    assert np.max(np.abs(other_swept_sine.impulse_response)) != pytest.approx(1)


# def test_nth_harmonic_time_delay():

# def test_nth_harmonic_sample_delay():


def test_sweep_frequency_at_time():
    """Verify that the instantaneous sweep frequency can be calculated for a given time."""
    f1 = 100
    swept_sine = SweptSine(20000, f1, 1000, 4)
    delta_t_f0 = swept_sine.nth_harmonic_time_delay(1)
    assert swept_sine.sweep_frequency_at_time(delta_t_f0) == pytest.approx(f1)
    delta_t_f3 = swept_sine.nth_harmonic_time_delay(3)
    assert swept_sine.sweep_frequency_at_time(delta_t_f3) == pytest.approx(f1 * 3)


# def test_get_fundamental_impulse_response():

# def test_get_harmonic_impulse_response():
