# Python Swept Sine

A python implementation of the synchronised swept sine based on [Novak et al. 2015](https://ant-novak.com/posts/research/2015-10-30_JAES_Swept/) and [Farina 2000](https://www.melaudia.net/zdoc/sweepSine.PDF).

## Getting started

```python
from swept_sine import SweptSine

my_swept_sine = SweptSine(
    fs=48000, f1=20, f2=20000, duration=10
    scale_dBFS=-6, # eg, to adjust the sweep amplitude at source
    fade_in=0.5, fade_out=0.5, fade_shape="cosine", # eg, to reduce ringing
    pad_start=1, pad_end=10 # eg, to account for latency and decay time
)

measurement = sd.playrec(my_swept_sine.sweep, my_swept_sine.fs, channels=1, blocking=True)

impulse_response = my_swept_sine.deconvolve(measurement)
```

During the init the sweep parameters are calculated and the sweep (`my_swept_sine.sweep`) and inverse filter (`my_swept_sine.inverse`) signals are generated. This includes calculating the actual sweep duration (`my_swept_sine.actual_duration`) required for phase synchronicity (see Novak et al. 2015, eq. 48).

## Measuring using .wav (for example in a DAW)

So that the signal generation and analysis can happen separately, the generated sweep can be saved to a wav file.

```python
my_swept_sine.save_sweep_as_wav(path="sweeps", prefix="my_sweep_name")

# creates "sweeps/my_sweep_name-params_48000_20_20000_10_-6_0.5_0.5_cosine_1_10.wav"
```

The sweep parameters are encoded into the filename so an instance of an identical sweep, including the computed parameters and inverse filter, can be re-created automatically from the file. Measurements can also be deconvolved directly from file.

```python
new_swept_sine = SweptSine.init_from_sweep_wav(
    "sweeps/my_sweep_name-params_params_48000_20_20000_10_-6_0.5_0.5_cosine_1_10.wav"
)

impulse_response = new_swept_sine.deconvolve_from_wav("measurement.wav")
```