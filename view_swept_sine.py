import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from swept_sine import SweptSine

swept_sine = SweptSine(
    fs=20000,
    f1=100,
    f2=1000,
    duration=5,
    # sweep_dBFS=-9,
    # fade_in=1,
    # fade_out=1,
    # fade_shape="cosine",
    # pad_start=1,
    # pad_end=1,
)

params = f"fs={swept_sine.fs}, f1={swept_sine.f1}, f2={swept_sine.f2}, duration={swept_sine.actual_duration} sweep_dBFS={swept_sine.sweep_dBFS}, fade_in={swept_sine.fade_in}, fade_out={swept_sine.fade_out}, fade_shape={swept_sine.fade_shape} pad_start={swept_sine.pad_start}, pad_end={swept_sine.pad_end}"

#### Time domain view of the generated signals

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))

ax1.plot(swept_sine.sweep)
ax1.set_title("Sweep")

ax2.plot(swept_sine.inverse)
ax2.set_title("Inverse")

impulse_response = swept_sine.deconvolve(swept_sine.sweep)

ax3.plot(impulse_response)
ax3.set_title("Impulse Response")

for ax in [ax1, ax2, ax3]:
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude (unit)")
    ax.grid()

normalise_tolerance_dB = 0.1
freqs, mag_dB = swept_sine.frequency_response(
    impulse_response, normalise_tolerance_dB=normalise_tolerance_dB
)
ax4.semilogx(
    freqs, mag_dB, label=f"Response (normalised +/-{normalise_tolerance_dB}dB)"
)

passband_tolerance_dB = 0.5
lower, upper = swept_sine.sweep_passband(tolerance_dB=passband_tolerance_dB)
passband_mask = (freqs >= lower) & (freqs <= upper)
ax4.semilogx(
    freqs[passband_mask],
    mag_dB[passband_mask],
    color="green",
    label=f"Passband (+/-{passband_tolerance_dB}dB)",
)

ax4.set_title("Frequency Response")

ax4.xaxis.set_major_formatter(ScalarFormatter())
ax4.ticklabel_format(style="plain", axis="x")
ax4.set_xlim(10, swept_sine.fs / 2)
ax4.set_ylim(-125, 10)
ax4.grid()
ax4.legend()

title = f"Generated Signals\n\n{params}"
fig.suptitle(title)
fig.tight_layout()


#### Phase Synchronicity View

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

window_length = swept_sine.fs // 20

pad_start_length = swept_sine._seconds_to_samples(swept_sine.pad_start, swept_sine.fs)
pad_end_length = swept_sine._seconds_to_samples(swept_sine.pad_end, swept_sine.fs)

for i, ax in enumerate(axes):
    n = i + 1
    idx = swept_sine.nth_harmonic_sample_delay(n)
    idx += pad_start_length
    if i == 5:
        if pad_end_length:
            ax.plot(swept_sine.sweep[-pad_end_length - window_length : -pad_end_length])
        else:
            ax.plot(swept_sine.sweep[-window_length:])
        ax.set_title(f"End of signal")
    else:
        ax.plot(swept_sine.sweep[idx : idx + window_length])
        if i == 0:
            ax.plot(
                swept_sine.sweep[pad_start_length : pad_start_length + window_length],
                ":",
                label="Start of signal",
            )
            ax.legend()
        f = swept_sine.sweep_frequency_at_time(swept_sine.nth_harmonic_time_delay(n))
        ax.set_title(f"Start of harmonic n={n} ({f:.2f}Hz)")
    ax.grid()

fig.suptitle("Phase Synchronicity View")
fig.tight_layout()

plt.show()
