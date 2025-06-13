import matplotlib.pyplot as plt
import sounddevice as sd

from swept_sine import SweptSine

my_swept_sine = SweptSine(fs=48000, f1=20, f2=20000, duration=5)

print("Measuring using the default audio device... ", end="", flush=True)
measurement = sd.playrec(
    my_swept_sine.sweep, samplerate=my_swept_sine.fs, channels=1, blocking=True
)
print("Done")

impulse_response = my_swept_sine.deconvolve(measurement)

plt.plot(impulse_response)
plt.title("Example Impulse Response")
plt.show()
