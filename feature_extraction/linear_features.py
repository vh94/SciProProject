import numpy as np
import pandas as pd
from scipy import signal, stats, integrate
import pywt

# Define frequency bands
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma1': (30, 47),
    'gamma2': (53, 75),
    'gamma3': (75, 97),
    'gamma4': (103, 128)
}

def compute_frequency_features(psds, freqs, channel_names, bands = bands ):
    """
    Compute EEG frequency domain features for seizure prediction.
    These are
    Args:
        psds: (n_epochs, n_channels, n_freqs) - PSD data
        freqs: (n_freqs,) - Frequency vector
        channel_names: list of channel names

    Returns:
        DataFrame with frequency features for all epochs and channels
    """
    n_epochs, n_channels, n_freqs = psds.shape

    freq_features_list = []

    for epoch_idx in range(n_epochs):
        epoch_dict = {}

        for ch_idx, ch_name in enumerate(channel_names):
            psd = psds[epoch_idx, ch_idx, :]

            # Total power using trapezoidal integration
            #total_power = np.trapz(np.abs(psd), freqs)
            total_power = integrate.trapezoid(np.abs(psd), freqs)

            # Compute absolute and relative band powers
            band_powers_abs = {}
            band_powers_rel = {}

            for band_name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs < high)
                band_power_abs = integrate.trapezoid(np.abs(psd[idx]), freqs[idx])
                band_power_rel = band_power_abs / total_power if total_power != 0 else 0

                band_powers_abs[band_name] = band_power_abs
                band_powers_rel[band_name] = band_power_rel

                # Store absolute power
                epoch_dict[f'{ch_name}_{band_name}_power'] = band_power_abs

            # Store relative powers
            for band_name in bands.keys():
                epoch_dict[f'{ch_name}_relative_{band_name}_power'] = band_powers_rel[band_name]

            # Total power
            epoch_dict[f'{ch_name}_total_power'] = total_power

            # Alpha peak frequency
            alpha_low, alpha_high = bands['alpha']
            alpha_idx = np.logical_and(freqs >= alpha_low, freqs < alpha_high)
            alpha_psd = psd[alpha_idx]
            alpha_freqs = freqs[alpha_idx]
            if len(alpha_psd) > 0:
                alpha_peak_idx = np.argmax(alpha_psd)
                alpha_peak_freq = alpha_freqs[alpha_peak_idx]
            else:
                alpha_peak_freq = np.nan
            epoch_dict[f'{ch_name}_alpha_peak_frequency'] = alpha_peak_freq

            # Mean frequency
            mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) != 0 else 0
            epoch_dict[f'{ch_name}_mean_frequency'] = mean_freq

            # Band power ratios - only first 7 bands (excluding gamma4)
            bands_for_ratios = ['delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2', 'gamma3']

            for i in range(len(bands_for_ratios)):
                for j in range(i + 1, len(bands_for_ratios)):
                    b1, b2 = bands_for_ratios[i], bands_for_ratios[j]
                    ratio = band_powers_abs[b1] / (band_powers_abs[b2] + 1e-10)
                    epoch_dict[f'{ch_name}_ratio_{b1}_{b2}'] = ratio

            # Special ratios
            beta_alpha_theta = band_powers_abs['beta'] / (band_powers_abs['alpha'] + band_powers_abs['theta'] + 1e-10)
            theta_alpha_beta = band_powers_abs['theta'] / (band_powers_abs['alpha'] + band_powers_abs['beta'] + 1e-10)
            epoch_dict[f'{ch_name}_ratio_beta_over_alpha_theta'] = beta_alpha_theta
            epoch_dict[f'{ch_name}_ratio_theta_over_alpha_beta'] = theta_alpha_beta

            # Spectral edge frequency and power at 50%
            cumsum_psd = np.cumsum(psd)
            sef50_idx = np.argmax(cumsum_psd >= 0.5 * cumsum_psd[-1]) if cumsum_psd[-1] > 0 else 0
            epoch_dict[f'{ch_name}_spectral_edge_frequency'] = freqs[sef50_idx]
            epoch_dict[f'{ch_name}_spectral_edge_power'] = cumsum_psd[sef50_idx]

        freq_features_list.append(epoch_dict)

    return pd.DataFrame(freq_features_list)


def univariate_linear_features(epochs):
    """
    Extract 59 univariate linear EEG features per channel
    Based on: "The goal of explaining black boxes in EEG seizure prediction is not to explain models' decisions"
    Published in Epilepsia Open (https://doi.org/10.1002/epi4.12748)

    Matches MATLAB implementation from:
    https://github.com/MauroSilvaPinto/Explaining-ML-EEG-Seizure-Prediction

    Input: epochs <class 'mne.epochs.Epochs'> (n_epochs, n_channels, n_times)
    Output: all_features <class 'pandas.DataFrame'> (n_epochs, n_channels x 59)
    """

    epochs_data = epochs.get_data()
    channel_names = epochs.ch_names
    # sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = epochs_data.shape

    # Compute PSD once for all epochs using MNE (more efficient!)
    epo_spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=128, n_fft=256)
    psds, freqs = epo_spectrum.get_data(return_freqs=True)  # Shape: (n_epochs, n_channels, n_freqs)

    # ========== FREQUENCY DOMAIN FEATURES (vectorized) ==========
    freq_features_df = compute_frequency_features(psds, freqs, channel_names)

    # ========== TIME DOMAIN & WAVELET FEATURES ==========
    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = {}

        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx]
            data = epochs_data[epoch_idx, ch_idx, :]

            # Statistical moments (use absolute values like MATLAB)
            abs_data = np.abs(data)
            epoch_features[f'{ch_name}_Normalized_mean_intensity'] = np.mean(abs_data) / np.max(abs_data)
            epoch_features[f'{ch_name}_Mean_intensity'] = np.mean(abs_data)
            epoch_features[f'{ch_name}_Std'] = np.std(abs_data)
            epoch_features[f'{ch_name}_Kurtosis'] = stats.kurtosis(data)
            epoch_features[f'{ch_name}_Skewness'] = stats.skew(data)

            # Hjorth parameters
            diff1 = np.diff(data)
            diff2 = np.diff(diff1)
            activity = np.var(data)
            mobility = np.std(diff1) / np.std(data)
            complexity = (np.std(diff2) / np.std(diff1)) / mobility

            epoch_features[f'{ch_name}_Activity'] = activity
            epoch_features[f'{ch_name}_Mobility'] = mobility
            epoch_features[f'{ch_name}_Complexity'] = complexity

            # Decorrelation time (matching MATLAB's xcorr 'unbiased' method)
            n = len(data)
            acf = np.correlate(data, data, mode='full') / n
            # Unbiased normalization: divide by (n - lag)
            lags = np.arange(-n+1, n)
            acf_unbiased = acf / (n - np.abs(lags))
            # Take only positive lags (like MATLAB: length(input_signal):end)
            acf_positive = acf_unbiased[n-1:]
            # Find first zero crossing (<=0)
            zero_cross = np.where(acf_positive <= 0)[0]
            decorr_idx = zero_cross[0] if len(zero_cross) > 0 else np.nan
            epoch_features[f'{ch_name}_decorrelation_time'] = decorr_idx
            # Wavelet features
            coeffs = pywt.wavedec(data, 'db4', level=5)
            # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]

            # Energy of detail coefficients D1-D5
            for d_idx in range(1, 6):
                coef = coeffs[6 - d_idx]  # Reverse order: cD1 is last
                energy = np.sum(coef ** 2)
                epoch_features[f'{ch_name}_energy_D{d_idx}'] = energy

            # Energy of approximation coefficient A5
            epoch_features[f'{ch_name}_energy_A5'] = np.sum(coeffs[0] ** 2)

        all_features.append(epoch_features)

    # Combine time domain and frequency domain features
    time_wavelet_df = pd.DataFrame(all_features)
    final_features = pd.concat([freq_features_df, time_wavelet_df], axis=1)

    return final_features


# Feature names matching MATLAB output order
linear_feature_names = [
    "Delta_power", "Theta_power", "Alpha_power", "Beta_power",
    "Gamma1_power", "Gamma2_power", "Gamma3_power", "Gamma4_power",
    "Relative_delta_power", "Relative_theta_power", "Relative_alpha_power",
    "Relative_beta_power", "Relative_gamma1_power", "Relative_gamma2_power",
    "Relative_gamma3_power", "Relative_gamma4_power", "Total_power",
    "Alpha_peak_frequency", "Mean_frequency",
    "Ratio_delta_theta", "Ratio_delta_alpha", "Ratio_delta_beta",
    "Ratio_delta_gamma1", "Ratio_delta_gamma2", "Ratio_delta_gamma3",
    "Ratio_theta_alpha", "Ratio_theta_beta", "Ratio_theta_gamma1",
    "Ratio_theta_gamma2", "Ratio_theta_gamma3",
    "Ratio_alpha_beta", "Ratio_alpha_gamma1", "Ratio_alpha_gamma2",
    "Ratio_alpha_gamma3",
    "Ratio_beta_gamma1", "Ratio_beta_gamma2", "Ratio_beta_gamma3",
    "Ratio_gamma1_gamma2", "Ratio_gamma1_gamma3", "Ratio_gamma2_gamma3",
    "Ratio_beta_over_alpha_theta", "Ratio_theta_over_alpha_beta",
    "Normalized_mean_intensity", "Mean_intensity", "Std", "Kurtosis",
    "Skewness", "Activity", "Mobility", "Complexity",
    "Spectral_edge_frequency", "Spectral_edge_power", "Decorrelation_time",
    "energy_D1", "energy_D2", "energy_D3", "energy_D4", "energy_D5", "energy_A5"
]

# Usage example:
# features_df = univariate_linear_features(epochs)
# print(f"Features shape: {features_df.shape}")
# print(f"Expected: ({len(epochs)}, {len(epochs.ch_names) * 59})")