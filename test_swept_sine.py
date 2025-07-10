import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from swept_sine import SweptSine


def test_init_creates_signals():
    """Verify that the class init creates the sweep and inverse signals."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    assert isinstance(swept_sine.sweep, np.ndarray)
    assert isinstance(swept_sine.inverse, np.ndarray)


def test_init_calculations_are_deterministic():
    """Verify that an instance can be re-created from its parameters/arguments."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    assert_array_equal(SweptSine(20000, 100, 1000, 4).sweep, swept_sine.sweep)


def test_init_from_sweep_filename(tmp_path):
    """Verify that an instance can be re-created from the parameters in a filename."""
    swept_sine = SweptSine(20000, 100, 1000, 4, -6, 1, 2, "linear", 2, 2)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="init_test")
    new_swept_sine = SweptSine.init_from_sweep_filename(filepath)
    assert all(
        getattr(swept_sine, key) == getattr(new_swept_sine, key)
        for key in SweptSine.re_init_parameters
    )
    assert_array_equal(new_swept_sine.sweep, swept_sine.sweep)


#### Utils


def test__enforce_2d_row_major():
    """Verify that audio data can be forced into 2D (samples, channels) shape."""
    assert SweptSine._enforce_2d_row_major(np.ones(48000)).shape == (
        48000,
        1,
    )  # 1D mono
    assert SweptSine._enforce_2d_row_major(np.ones((48000, 1))).shape == (
        48000,
        1,
    )  # 2D mono
    assert SweptSine._enforce_2d_row_major(np.ones((2, 48000))).shape == (
        48000,
        2,
    )  # 2D swapped
    assert SweptSine._enforce_2d_row_major(np.ones((48000, 2))).shape == (
        48000,
        2,
    )  # 2D correct
    with pytest.raises(ValueError):
        SweptSine._enforce_2d_row_major(np.ones((48000, 1, 1)))  # 3D


def test__scale_by_dBFS():
    """Verify that the sweep, and the resulting impulse response can be scaled by a dBFS value."""
    swept_sine = SweptSine(20000, 100, 1000, 4, sweep_dBFS=-6)
    assert np.max(np.abs(swept_sine.sweep)) == pytest.approx(0.5, abs=0.01)
    assert np.max(swept_sine.deconvolve(swept_sine.sweep)) == pytest.approx(
        0.5, abs=0.01
    )


def test__seconds_to_samples():
    """Check conversion from seconds to samples using sampling rate."""
    assert SweptSine._seconds_to_samples(0.5, 48000) == 24000


def test__samples_to_seconds():
    """Check conversion from a number of samples to seconds using sampling rate."""
    assert SweptSine._samples_to_seconds(24000, 48000) == 0.5


def test__fade_in_out():
    """Verify that a signal can be faded in and out."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    test_data = np.ones((20000, 1))  # 1 second of ones
    unfaded_test_data = swept_sine._fade_in_out(test_data, 0, 0, fade_shape="linear")
    assert_array_equal(unfaded_test_data, test_data)
    faded_test_data = swept_sine._fade_in_out(test_data, 0.5, 0.5, fade_shape="linear")
    assert faded_test_data[0, 0] == 0
    assert faded_test_data[10000, 0] == 1
    assert faded_test_data[-1, 0] == 0


data_test__create_fade_envelope = [
    (100, 0, 1, "linear"),
    (100, 1, 0, "linear"),
    (100, 0, 1, "cosine"),
    (100, 1, 0, "cosine"),
]


@pytest.mark.parametrize(
    "length, start, end, fade_shape", data_test__create_fade_envelope
)
def test__create_fade_envelope(length, start, end, fade_shape):
    """Verify drawing curves with defined start and end points"""
    curve = SweptSine._create_fade_envelope(length, start, end, fade_shape)
    assert curve[0] == start
    assert curve[length // 2] == pytest.approx(np.abs(start - end) / 2, 0.1)
    assert curve[-1] == end


def test__zero_pad_start_end():
    swept_sine = SweptSine(20000, 100, 1000, 4)
    test_data = np.ones((20000, 1))
    unpadded_test_data = swept_sine._zero_pad_start_end(test_data, 0, 0)
    assert_array_equal(unpadded_test_data, test_data)
    padded_test_data = swept_sine._zero_pad_start_end(test_data, 0.5, 2)
    assert padded_test_data.shape == (70000, 1)


#### WAV Files


def test__construct_wav_filepath():
    """Verify that a wav filename can be constructed including the sweep parameters."""
    from pathlib import Path

    swept_sine = SweptSine(20000, 100, 1000, 4, -6, 1, 2, "linear", 2, 2)
    filepath = swept_sine._construct_wav_filepath(path="test/path", prefix="my_sweep")
    assert isinstance(filepath, Path)
    assert filepath == Path(
        f"test/path/my_sweep-params_20000_100_1000_4_-6_1_2_linear_2_2.wav"
    )


def test__parameters_from_wav_filename():
    """Verify that the sweep parameters can be read from a properly formatted filename."""
    filepath = "test/path/my_sweep-params_20000_100_1000_4_-6_1_2_linear_2_2.wav"
    assert SweptSine._parameters_from_wav_filename(filepath) == [
        20000,
        100,
        1000,
        4,
        -6,
        1,
        2,
        "linear",
        2,
        2,
    ]


def test_save_sweep_as_wav_and__read_from_wav(tmp_path):
    """Verify that saving and reading from to f32 wav maintains the sweep signal format, etc."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="save_test")
    data = swept_sine._read_from_wav(filepath)  # enforces 2D (samples, channels) shape
    assert_allclose(data, swept_sine.sweep)


def test_deconvolve_from_wav(tmp_path):
    """Verify that measurements deconvolved directly and from a wav file are identical."""
    swept_sine = SweptSine(20000, 100, 1000, 4)
    filepath = swept_sine.save_sweep_as_wav(path=tmp_path, prefix="deconvolve_test")
    wav_impulse_response = swept_sine.deconvolve_from_wav(filepath)
    direct_impulse_response = swept_sine.deconvolve(swept_sine.sweep)
    # values will be small so only check the absolute difference is within f32 precision
    assert_allclose(wav_impulse_response, direct_impulse_response, rtol=0, atol=1e-7)


#### Sweep Parameters

# def test__calculate_L():

# def test__calculate_T():

# def test__generate_t():

# def test__generate_sweep():

# def test__generate_inverse():
# As per Farina 2000: 20*log10(_inverse_envelope(t, L))[-1] == -6*np.log2(f2/f1)


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


def test_sweep_frequency_at_time():
    """Verify that the instantaneous sweep frequency can be calculated for a given time."""
    f1 = 100
    swept_sine = SweptSine(20000, f1, 1000, 4)
    delta_t_f0 = swept_sine.nth_harmonic_time_delay(1)
    assert swept_sine.sweep_frequency_at_time(delta_t_f0) == pytest.approx(f1)
    delta_t_f3 = swept_sine.nth_harmonic_time_delay(3)
    assert swept_sine.sweep_frequency_at_time(delta_t_f3) == pytest.approx(f1 * 3)


def test_get_bin_idx_closest_to_f():
    freqs = np.array([2, 4, 6, 8, 10])
    idx = SweptSine.get_bin_idx_closest_to_f(freqs, 7)
    assert idx == 2  # the bin with 6 in it


# def test_sweep_passband():


def test_deconvolution_N():
    swept_sine = SweptSine(20000, 10, 500, 1)
    assert len(swept_sine.sweep) == len(swept_sine.inverse) == len(swept_sine._t)
    assert swept_sine.deconvolution_N == len(swept_sine._t) + len(swept_sine._t) - 1


def test_deconvolve():
    """Verify (only) that the normalisation of the deconvolution is correct."""
    swept_sine = SweptSine(20000, 10, 500, 1)
    reference_impulse_response = swept_sine.deconvolve(swept_sine.sweep)
    assert reference_impulse_response.shape == (swept_sine.deconvolution_N, 1)
    assert np.max(np.abs(reference_impulse_response)) == pytest.approx(1)
    attenuated_impulse_response = swept_sine.deconvolve(swept_sine.sweep * 0.5)
    assert np.max(np.abs(attenuated_impulse_response)) == pytest.approx(0.5)


# def test_frequency_response():

# def test_nth_harmonic_time_delay():

# def test_nth_harmonic_sample_delay():

# def test_get_fundamental_impulse_response():

# def test_get_harmonic_impulse_response():


#### Convolve


def test_convolve():
    """Verify the convolution output shape and response to dirac."""
    rng = np.random.default_rng()
    data = rng.uniform(-1, 1, 20000)
    dirac = [1] + [0] * 999  # len=1000
    convolution = SweptSine.convolve(data, dirac)
    assert convolution.ndim == 2
    assert len(convolution) == 20999
    assert_allclose(convolution[: len(data), 0], data)
    next_fast_len = SweptSine.convolve([1], [0], N=1234, next_fast_len=True)
    assert len(next_fast_len) == 1250
