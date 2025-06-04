# Python Swept Sine

A python implementation of the synchronised swept sine based on [Novak et al. 2015](https://ant-novak.com/posts/research/2015-10-30_JAES_Swept/) and [Farina 2000](https://www.melaudia.net/zdoc/sweepSine.PDF).


## Example of a measurement in Python

    import sounddevice as sd

    my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

    measurement = sd.playrec(my_swept_sine.sweep, my_swept_sine.fs)

    impulse_response = my_swept_sine.deconvolve(measurement)


## Example of a measurement elsewhere (eg in a DAW)

Create the `SweptSine` instance as normal then save the sweep signal to a wav file using `save_sweep_as_wav()`. This will automatically append the sweep parameters to the filename for reference during devonvolution later.

    my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

    my_swept_sine.save_sweep_as_wav(path="sweeps", prefix="my_sweep_name")

    # writes the file: "sweeps/my_sweep_name-48000-20-20000-5.wav"

The sweep signal can be used for measurements elsewhere (eg in a DAW), and then devonvolved by referencing a `SweptSine` instance with the same parameters.

    my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

    impulse_response = my_swept_sine.deconvolve_from_wav("my_measurement.wav")

### Creating a `SweptSine` instance from the sweep filename

For convenience, `SweptSine` can be instantiated from a sweep filename using `init_from_sweep_wav()` which extracts the sweep parameters from filenames created with `save_sweep_as_wav()`.

    my_swept_sine = SweptSine.init_from_sweep_wav("my_sweep_name-48000-20-20000-5.wav")

is equivalent to

    my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)
