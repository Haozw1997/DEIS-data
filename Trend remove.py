import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency and frequency ranges
Fs = 4000
Range = [1, 10, 100]
frequencies = {
    1: np.round([1, 1.3, 1.7, 2.2, 2.8, 3.6, 4.6, 6, 7.7, 10], 1),
    10: np.round([10, 13, 17, 22, 28, 36, 46, 60, 77, 100], 0),
    100: np.round([100, 130, 170, 220, 280, 360, 460, 600, 770, 1000], -1)
}
lengths = {1: 1000, 10: 100, 100: 10}
filenames = {
    1: 'C-DEIS_1-10Hz.txt',
    10: 'C-DEIS_10-100Hz.txt',
    100: 'C-DEIS_100-1000Hz.txt'
}

# Load data files
Tol = {key: np.loadtxt(fname) for key, fname in filenames.items()}
CalZ = {}

# Process for mode=3 (Raw) and mode=4 (Linear fitting)
for mode in [2, 3]:  # Column index: 2 = column 3 in MATLAB, 3 = column 4
    result = {}
    for r in Range:
        data = Tol[r]
        freqs = frequencies[r]
        cut_len = lengths[r]
        data = data[:-cut_len]  # Remove the trailing part of the signal

        # Determine analysis window: take last 10 periods of excitation
        stp = len(data) - int(10 * Fs * (1 / r)) + 1
        Fit = np.zeros((len(data) - stp, 4))
        Fit[:, 0] = data[stp:, 0] - data[stp, 0]  # Time normalization
        Fit[:, 1] = data[stp:, 1]                 # Current signal
        Fit[:, 2] = data[stp:, 2]                 # Voltage signal

        # Linear trend removal for voltage
        if mode == 3:
            p = np.polyfit(Fit[:, 0], Fit[:, 2], 1)
            trend = np.polyval(p, Fit[:, 0])
            Fit[:, 3] = Fit[:, 2] - trend         # Linear fitting result
        else:
            Fit[:, 3] = Fit[:, 2]                 # Raw data without trend removal

        # FFT for voltage
        V = Fit[:, mode]
        I = Fit[:, 1]
        N = len(V)
        fft_V = np.fft.fft(V) / N
        fft_I = np.fft.fft(I) / N
        ff = Fs / N * np.arange(N // 2 + 1)

        # Find the FFT index corresponding to each frequency
        pos = [np.argmin(np.abs(ff - f)) for f in freqs]

        # Calculate impedance
        Z_re = np.real(fft_V[pos] / fft_I[pos])
        Z_im = np.imag(fft_V[pos] / fft_I[pos])
        Z_cal = np.column_stack((freqs, Z_re, Z_im))
        result[r] = Z_cal

    # Combine frequency segments
    all_data = np.vstack([result[1], result[10], result[100]])
    x = all_data[:, 0]
    y1 = all_data[:, 1]
    y2 = all_data[:, 2]
    unique_x = np.unique(x)

    avg_data = np.zeros((len(unique_x), 3))
    for i, ux in enumerate(unique_x):
        idx = x == ux
        avg_data[i, 0] = ux
        avg_data[i, 1] = np.mean(y1[idx])
        avg_data[i, 2] = np.mean(y2[idx])

    CalZ[f'mode{mode+1}'] = {'totDEIS': avg_data}

# Plot Nyquist diagram
plt.figure(figsize=(6, 5))
plt.plot(CalZ['mode3']['totDEIS'][:, 1], -CalZ['mode3']['totDEIS'][:, 2],
         label='Raw Data', linewidth=1.5)
plt.plot(CalZ['mode4']['totDEIS'][:, 1], -CalZ['mode4']['totDEIS'][:, 2],
         label='Linear Fitting', linewidth=1.5)
plt.xlabel('Re(Z) / Ω')
plt.ylabel('-Im(Z) / Ω')
plt.title('DEIS Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
