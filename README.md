# Python Swept Sine

A python implementation of the synchronised swept sine based on [Novak et al. 2015](https://ant-novak.com/posts/research/2015-10-30_JAES_Swept/) and [Farina 2000](https://www.melaudia.net/zdoc/sweepSine.PDF).


## Parameters

Required:

- `fs`: Sampling rate
- `f1`: Starting (lower) frequency in Hz
- `f2`: Ending (higher) frequency in HZ
- `duration`: Desired duration of the sweep in seconds

Optional:

- `sweep_dBFS`: Amplitude of the sweep in dBFS. Default=0
- `fade_in`: Duration of a fade in in seconds. Default=0
- `fade_out`: Duration of a fade out in seconds. Default=0
- `fade_shape`: Shape of the fade in/out curve. One of: `"linear"` or `"cosine"`. Default="cosine".
- `pad_start`: Duration of zero padding at the start in seconds. Default=0
- `pad_end`: Duration of zero padding at the end in seconds. Default=0

Fading can be helpful to manage ringing in resulting impulse responses. Padding can be helpful to account for measurement latency and to capture the full reverberant decay.

### Duration

For a synchronised swept sine, the actual duration of the sweep will set based on the target `duration` argument and the rate of change required between `f1` and `f2` for phase synchronicity. Therefore, `duration` (which sets the property `target_duration`) will not necessarily match the property `actual_duration` and the length of the signals, as shown below:

```
>>> from swept_sine import SweptSine
>>>
>>> my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)
>>>
>>> my_swept_sine.target_duration
5
>>>
>>> my_swept_sine.actual_duration
np.float64(4.835428695287496)
>>>
>>> my_swept_sine.sweep.shape
(232101, 1)
>>>
>>> my_swept_sine.sweep.shape[0] / my_swept_sine.fs
4.8354375
```

## Example of a measurement in Python

```python
import sounddevice as sd

my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

measurement = sd.playrec(my_swept_sine.sweep, my_swept_sine.fs, channels=1, blocking=True)

impulse_response = my_swept_sine.deconvolve(measurement)
```

## Example of a measurement elsewhere (eg in a DAW)

Create the `SweptSine` instance as normal then save the sweep signal to a wav file using `save_sweep_as_wav()`. This will automatically append the sweep parameters (including the optional/defaults) to the filename for reference during devonvolution later.

```python
my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

my_swept_sine.save_sweep_as_wav(path="sweeps", prefix="my_sweep_name")

# writes a file called "sweeps/my_sweep_name-params_20000_100_1000_4_0_0_0_cosine_0_0.wav"
```

The sweep signal can be used for measurements elsewhere (eg in a DAW), and then devonvolved by referencing a `SweptSine` instance with the same parameters.

```python

# when the measurement "my_measurement.wav" has been created elsewhere

my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

impulse_response = my_swept_sine.deconvolve_from_wav("my_measurement.wav")
```

### Creating a `SweptSine` instance from the sweep filename

For convenience, `SweptSine` can be instantiated from a sweep filename using `init_from_sweep_wav()` which extracts the sweep parameters from filenames created with `save_sweep_as_wav()`.

```python
my_swept_sine = SweptSine.init_from_sweep_wav(
    "my_sweep_name-params_20000_100_1000_4_0_0_0_cosine_0_0.wav"
)
```

is equivalent to

```python
my_swept_sine = SweptSine(
    fs=48000, f1=20, f2=20000, duration=5,
    sweep_dBFS=0, fade_in=0, fade_out=0, fade_shape="cosine", pad_start=0, pad_end=0
)
```

So, if measurements have been made elsewhere then they can be processed directly from the wav files and their filenames:

```python
my_swept_sine = SweptSine.init_from_sweep_wav(
    "my_sweep_name-params_20000_100_1000_4_0_0_0_cosine_0_0.wav.wav"
)
impulse_response = my_swept_sine.deconvolve_from_wav("my_measurement.wav")
```