import matplotlib.pyplot as plt

from swept_sine import SweptSine

swept_sine = SweptSine(20000, 100, 1000, 4)

plt.figure()
plt.plot(swept_sine.sweep)
plt.plot(swept_sine.inverse)
plt.title("Sweep and Inverse Filter")

plt.figure()
plt.plot(swept_sine.deconvolve(swept_sine.sweep))
plt.title("Impulse Response")

plt.show()
