import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mne

print(f"mne version: {mne.__version__}")
# Load the EDF file
edf_file = '/Volumes/Extreme SSD/EEG_Databases/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-00_eeg.edf'  # Replace with the path to your EDF file
raw = mne.io.read_raw_edf(edf_file, preload=True)
mne.viz.plot_raw(raw)
# Apply a bandpass filter (optional)
raw.filter(l_freq=1, h_freq=40)
# Define the window length (5 seconds) and create epochs
epoch_length = 5  # 5 seconds
# Create epochs manually by dividing the raw data into non-overlapping 5-second chunks
# Each epoch will span from t=0 to t=5, t=5 to t=10, etc.
epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, overlap=0, preload=True)
# Compute the Power Spectral Density (PSD) using the 5-second epochs
print("calculation PSD")
epo_spectrum = epochs.compute_psd()
psds, freqs = epo_spectrum.get_data(return_freqs=True)


n_epochs = psds.shape[0]  # Number of epochs (525)
n_channels = psds.shape[1]  # Number of channels (19)
n_freqs = psds.shape[2]  # Number of frequency bins (641)

print(f"Number of epochs: {n_epochs}, Number of channels: {n_channels}, Number of frequencies: {n_freqs}")

n_features = n_channels * 59
print(f"Number of features: {n_features}")
# Set up the plot
plt.figure(figsize=(10, 6))
# Loop over each channel and plot its PSD
for i in range(n_channels):
    avg_psd = psds[i, :, :].mean(axis=0)
    plt.plot(freqs,  avg_psd , label=f'Channel {i+1}')

# Customize the plot
plt.title('Avg Power Spectral Density (PSD) for Each Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (uV^2/Hz)')
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), title='Channels')
plt.grid(True)
plt.tight_layout()

# Show the plot
# plt.show()


## Interactive VIZ
fig = go.Figure()

# Loop over each channel and epoch to create traces
for i in range(n_channels):
    for j in range(n_epochs):
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psds[j, i, :],  # PSD for channel i and epoch j
                mode='lines',
                name=f'Channel {i + 1} Epoch {j + 1}',
                visible=True if j == 0 else False  # Show only the first epoch initially
            )
        )

# Create slider steps
slider_steps = []
for j in range(n_epochs):
    # Create visibility list: True for current epoch, False for others
    visible = [False] * (n_channels * n_epochs)
    for i in range(n_channels):
        trace_index = j * n_channels + i  # Calculate the index of the trace
        visible[trace_index] = True

    slider_step = {
        'args': [
            {'visible': visible},  # Set visibility for all traces
            {'title': f'Power Spectral Density - Epoch {j + 1}'}
        ],
        'label': f'Epoch {j + 1}',
        'method': 'update'
    }
    slider_steps.append(slider_step)

# Define layout with slider
fig.update_layout(
    title="Power Spectral Density (PSD) for Each Channel (Interactive Epochs)",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Power Spectral Density (uV^2/Hz)",
    sliders=[{
        'active': 0,
        'currentvalue': {'prefix': 'Epoch: ', 'visible': True},
        'pad': {'b': 10},
        'steps': slider_steps
    }],
    showlegend=True,
    xaxis = dict(range=[0.0, 40.0])
)

# Show the figure
fig.show()
