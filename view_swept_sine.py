import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from swept_sine import SweptSine

swept_sine = SweptSine(fs=20000, f1=100, f2=1000, duration=5)


swept_sine.deconvolve(swept_sine.sweep)  # writes to self.impulse_response


#### Time Domain Signal View

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9))

ax1.plot(swept_sine.sweep)
ax1.set_title("Sweep")

ax2.plot(swept_sine.inverse)
ax2.set_title("Inverse")

ax3.plot(swept_sine.impulse_response)
ax3.set_title("Full Impulse Response")

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (unit)")
    ax.grid()

fig.suptitle("Time Domain Signal View")
fig.tight_layout()


#### Frequency Domain Signal View

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.semilogx(*swept_sine.to_dBFS(swept_sine.sweep), label="Sweep")
ax.semilogx(*swept_sine.to_dBFS(swept_sine.inverse), label="Inverse")
ax.semilogx(*swept_sine.to_dBFS(swept_sine.impulse_response), label="Impulse Response")

ax.set_xlim(20, 20000)
ax.set_ylim(-120, 10)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style="plain", axis="x")
ax.legend()

fig.suptitle("Frequency Domain Signal View")
fig.tight_layout()


#### Phase Synchronicity View

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes = axes.flatten()

window_length = swept_sine.fs // 20

for i, ax in enumerate(axes):
    n = i + 1
    idx = swept_sine.nth_harmonic_sample_delay(n)
    if i == 5:
        ax.plot(swept_sine.sweep[-window_length:])
        ax.set_title(f"End of signal")
    else:
        ax.plot(swept_sine.sweep[idx : idx + window_length])
        if i == 0:
            ax.plot(swept_sine.sweep[:window_length], ":", label="Start of signal")
            ax.legend()
        f = swept_sine.sweep_frequency_at_time(swept_sine.nth_harmonic_time_delay(n))
        ax.set_title(f"Start of harmonic n={n} ({f:.2f}Hz)")
    ax.grid()

fig.suptitle("Phase Synchronicity View")
fig.tight_layout()

plt.show()
