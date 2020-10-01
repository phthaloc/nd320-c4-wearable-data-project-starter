import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('ggplot')


__author__ = 'Udo Dehm, udacity'
__copyright__ = 'Copyright 2020'
__credits__ = ['Udo Dehm', 'udacity']
__license__ = ''
__version__ = '0.1.0'
__maintainer__ = 'Udo Dehm'
__email__ = 'udo.dehm@mailbox.org'
__status__ = 'Dev'


def plot_raw_signals(ppg, accx, accy, accz, fs):
    """
    Plot all raw signals that belong together
    :param ppg: PPG signal
    :param accx: Accelerometer signal in x-direction
    :param accy: accelerometer signal in y-direction
    :param accz: accelerometer signal in z-direction
    :return: figure with all signals
    """
    ts = np.arange(0, len(ppg), 1) / fs
    fig, axarr = plt.subplots(4, figsize=(15, 15))
    axarr[0].plot(ts, ppg)
    axarr[0].set_xlabel('time (sec)')
    axarr[0].set_ylabel('amplitude')
    axarr[0].set_title('PPG signal')
    axarr[1].plot(ts, accx)
    axarr[1].set_xlabel('time (sec)')
    axarr[1].set_ylabel('amplitude (mV)')
    axarr[1].set_title('Acceleration x-axis signal')
    axarr[2].plot(ts, accy)
    axarr[2].set_xlabel('time (sec)')
    axarr[2].set_ylabel('amplitude (mV)')
    axarr[2].set_title('Acceleration y-axis signal')
    axarr[3].plot(ts, accz)
    axarr[3].set_xlabel('time (sec)')
    axarr[3].set_ylabel('amplitude (mV)')
    axarr[3].set_title('Acceleration z-axis signal')
    plt.tight_layout()
    return fig


def plot_spectogram(ppg, accx, accy, accz, labels, fs, window_length, window_shift, min_freq, max_freq):
    """
    plot sectrograms of all PPG and accelerometer signals
    :param ppg: PPG signal
    :param accx: Accelerometer signal in x-direction
    :param accy: accelerometer signal in y-direction
    :param accz: accelerometer signal in z-direction
    :param labels: array with labels (heart beat in Hz)
    :param fs: sample frequency
    :param window_length: length of the window/filter (in sample units)
    :param window_shift: shift of window for each step/ stride (in sample units)
    :param min_freq: minimum frequency (y-axis) to plot
    :param max_freq: maximum frequency (y-axis) to plot
    :return: sectrogram figure object
    """
    ts_center = np.arange(0, len(labels), 1) * window_shift / fs + window_length / fs / 2

    # timestamps in minutes
    fig, axarr = plt.subplots(4, figsize=(15, 20))
    # we are splitting data x into NFFT length segments (local/smaller time segments) and compute the spectrum 
    # (fourier transform) of each section.
    # freq contains the frequencies containing to the rows (y-axis ^= frequencies) in the spectrum
    # Fs: sample frequency
    # NFFT: number of data points in each segment -> we will have ca. len(x)/NFFT segments 
    # (=time segments = rows in spec matrix). 
    # This is only approximately because we most likely divide the signel by a number that is an exact divisor of len(x)
    # here: we have a sample frequency of fs Hz and calculate FFT in intervals of length NFFT=fs*10 (= 10 sec).
    ppg_spec, ppg_spec_freqs, ppg_t, ppg_img = axarr[0].specgram(
        x=ppg, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[0].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[0].set_xlabel('time (sec)')
    axarr[0].set_ylabel('frequency (Hz)')
    axarr[0].set_title('Spectogram of PPG signal')
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ppg_img, cax=cax, orientation='vertical')
    ppg_img.set_clim(0, 50)
    axarr[0].set_ylim((min_freq, max_freq))
    axarr[0].legend()

    
    accx_spec, accx_spec_freqs, accx_t, accx_img = axarr[1].specgram(
        x=accx, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[1].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[1].set_xlabel('time (sec)')
    axarr[1].set_ylabel('frequency (Hz)')
    axarr[1].set_title('Spectogram of acceleration signal in x-direction')
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(accx_img, cax=cax, orientation='vertical')
    accx_img.set_clim(-50, 0)
    axarr[1].set_ylim((min_freq, max_freq))
    axarr[1].legend()

    accy_spec, accy_spec_freqs, accy_t, accy_img = axarr[2].specgram(
        x=accy, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[2].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[2].set_xlabel('time (sec)')
    axarr[2].set_ylabel('frequency (Hz)')
    axarr[2].set_title('Spectogram of acceleration signal in y-direction')
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(accy_img, cax=cax, orientation='vertical')
    accy_img.set_clim(-50, 0)
    axarr[2].set_ylim((min_freq, max_freq))
    axarr[2].legend()

    accz_spec, accz_spec_freqs, accz_t, accz_img = axarr[3].specgram(
        x=accz, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[3].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[3].set_xlabel('time (sec)')
    axarr[3].set_ylabel('frequency (Hz)')
    axarr[3].set_title('Spectogram of acceleration signal in z-direction')
    divider = make_axes_locatable(axarr[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(accz_img, cax=cax, orientation='vertical')
    accz_img.set_clim(-50, 0)
    axarr[3].set_ylim((min_freq, max_freq))
    axarr[3].legend()

    plt.tight_layout()
    return fig


def plot_spectogram_sumarized(ppg, acc, labels, fs, window_length, window_shift, min_freq, max_freq):
    """
    plot sectrograms of all PPG and accelerometer signals
    :param ppg: PPG signal
    :param accx: Accelerometer signal in x-direction
    :param accy: accelerometer signal in y-direction
    :param accz: accelerometer signal in z-direction
    :param labels: array with labels (heart beat in Hz)
    :param fs: sample frequency
    :param window_length: length of the window/filter (in sample units)
    :param window_shift: shift of window for each step/ stride (in sample units)
    :param min_freq: minimum frequency (y-axis) to plot
    :param max_freq: maximum frequency (y-axis) to plot
    :return: sectrogram figure object
    """
    ts_center = np.arange(0, len(labels), 1) * window_shift / fs + window_length / fs / 2

    # timestamps in minutes
    fig, axarr = plt.subplots(2, figsize=(15, 12))
    # we are splitting data x into NFFT length segments (local/smaller time segments) and compute the spectrum 
    # (fourier transform) of each section.
    # freq contains the frequencies containing to the rows (y-axis ^= frequencies) in the spectrum
    # Fs: sample frequency
    # NFFT: number of data points in each segment -> we will have ca. len(x)/NFFT segments 
    # (=time segments = rows in spec matrix). 
    # This is only approximately because we most likely divide the signel by a number that is an exact divisor of len(x)
    # here: we have a sample frequency of fs Hz and calculate FFT in intervals of length NFFT=fs*10 (= 10 sec).
    ppg_spec, ppg_spec_freqs, ppg_t, ppg_img = axarr[0].specgram(
        x=ppg, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[0].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[0].set_xlabel('time (sec)')
    axarr[0].set_ylabel('frequency (Hz)')
    axarr[0].set_title('Spectogram of PPG signal')
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ppg_img, cax=cax, orientation='vertical')
    ppg_img.set_clim(0, 50)
    axarr[0].set_ylim((min_freq, max_freq))
    axarr[0].legend()

    
    accx_spec, accx_spec_freqs, accx_t, accx_img = axarr[1].specgram(
        x=acc, NFFT=window_length, Fs=fs,
        noverlap=window_length-window_shift, cmap='jet_r'
    )
    axarr[1].plot(ts_center, labels, 'o', markersize=8, color='k', label='label')
    axarr[1].set_xlabel('time (sec)')
    axarr[1].set_ylabel('frequency (Hz)')
    axarr[1].set_title('Spectogram of acceleration signal in x-direction')
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(accx_img, cax=cax, orientation='vertical')
    accx_img.set_clim(-50, 0)
    axarr[1].set_ylim((min_freq, max_freq))
    axarr[1].legend()

    plt.tight_layout()
    return fig


def plot_window_prediction (
    freqs,
    ppg_fft,
    ppg_peaks_freqs,
    ppg_peaks_pwr,
    ppg_label,
    acc_fft,
    accx_fft,
    accy_fft,
    accz_fft,
    acc_peaks_freqs,
    acc_peaks_pwr,
    min_freq,
    max_freq
):
    """
    Plot a window of the complete signal which is used for feature extraction.
    :param freqs: frequencies that belong to the FFT transforms
    :param ppg_fft: (fast) fourier transform of the PPG signal
    :param ppg_peaks: list of frequencies indicating the peaks of the ppg_fft signal
    :param ppg_label: (float) PPG signal label (heart beat in BPM for the window sequence)
    :param acc_fft: (fast) fourier transform of total accelerometer signal
    :param accx_fft: (fast) fourier transform of accelerometer signal in x-direction
    :param accy_fft: (fast) fourier transform of accelerometer signal in y-direction
    :param accz_fft: (fast) fourier transform of accelerometer signal in z-direction
    :param acc_peaks: list of peaks indicating the peaks of the acc_fft signal
    :param min_freq: minimum frequency (x-axis) to plot
    :param max_freq: maximum frequency (x-axis) to plot
    :return: figure with all given relevant signals
    """
    fig, ax = plt.subplots(1, figsize=(15, 6))
    color = 'tab:red'
    ax.plot(freqs, ppg_fft, '.-', color=color, label='fft of PPG')
    ax.plot(ppg_peaks_freqs, ppg_peaks_pwr, '.', color='b', markersize=10, label='found peaks')
    ax.axvline(ppg_label/60, color='k', label='true heart rate (label)')
    ax.axvline(min_freq, color='0.5', label='min. frequency')
    ax.axvline(max_freq, color='0.5', label='max. frequency')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('PPG FFT power amplitude')
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = 'tab:green'
    ax2.plot(freqs, acc_fft, '.-', color=color, label='fft of ACC')
    # trick for adding label of ax2 in legend:
    ax.plot([], [], '.-', color=color, label='fft of ACC')
    ax2.plot(acc_peaks_freqs, acc_peaks_pwr, '.', color='b', markersize=10)
    ax2.plot(freqs, accx_fft, '.--', color='0.7', label='fft of ACC x, y and z direction')
    ax2.plot(freqs, accy_fft, '.-.', color='0.7', label='fft of ACC x, y and z direction')
    ax2.plot(freqs, accz_fft, '.-', color='0.7', label='fft of ACC x, y and z direction')
    ax.plot(freqs, accz_fft, '.-', color='0.7', label='fft of ACC x, y and z direction')
    ax2.set_ylabel('ACC FFT power amplitude')
    ax2.tick_params(axis='y', labelcolor=color)

    ax.set_xlim(0.9*min_freq, max_freq*1.05)
    ax.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return fig
