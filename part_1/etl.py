import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('ggplot')


__author__ = 'Udo Dehm, udacity'
__credits__ = ['Udo Dehm', 'udacity']
__license__ = ''
__version__ = '0.1.0'
__maintainer__ = 'Udo Dehm'
__email__ = 'udo.dehm@mailbox.org'
__status__ = 'Dev'


def LoadTroikaDataset():
    """
    Get filenames in considered data directory.
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data (labels)
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls


def load_troika_data(data_fl):
    """
    Load complete data file with all signals
    :return: dictionary with keys '__header__', '__version__', '__globals__', 'sig'.
        Most important is the signal key 'sig'.
        The sig key contains a matrix with 6 rows:
            row 1: ECG signal (can be used to calculate the reference heart rate signal -> already done, see REF_... files)
            row 2: 1st PPG channel
            row 3: 2nd PPG channel
            row 4: x-axis of acceleration data
            row 5: y-axis of acceleration data
            row 6: z-axis of acceleration data
    """
    return sp.io.loadmat(data_fl)


def LoadTroikaDataFile(data_fl):
    """
    Load data.
    Loads and extracts signals from a troika data file.
    We are only insterested in these signals: one PPG channel (we use the 2nd), the three acceleration data channels.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = load_troika_data(data_fl)['sig']
    return data[2:]


def load_labels(ref_fl):
    """
    load reference signal (label)
    The reference signal contains the ground-truth of heart rate in beats per minute (BPM)
    in an 8 second window. Two consecutive windows are shifted by 2 seconds. This means
    that consecutive windows are overlapping by 6 seconds. First window covers the first 8
    seconds, second window covers 3rd second to 10th second, ...    
    """
    return sp.io.loadmat(ref_fl)['BPM0'].flatten()


def data_information(file_data, file_label, fs, window_shift_sec, window_length_sec):
    """
    Print information about files containing the data.
    :param file_data: list of file names containing the data to import
    :param file_label: list of file names containing the labels of the data
    :param fs: Sampling frequency of imported time series
    :param window_shift_sec: shift of window for each step/stride (in seconds)
    :param window_length_sec: length of the window/filter (in seconds)
    """
    for i, data_fl in enumerate(file_data):
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
        ppg_ref = load_labels(file_label[i])
        print(f"signal length of {data_fl.split('/')[-1]} = {len(ppg)/fs} sec ({len(ppg)} data points)")
        print(f'ppg signal min: {ppg.min()}')
        print(f'ppg signal max: {ppg.max()}')
        ts_upper = np.arange(0, len(ppg_ref), 1) * window_shift_sec + window_length_sec
        print(f"reference signal length (largest upper interval limit): {ts_upper[-1]} sec ({ts_upper[-1]*FS})")
        print(f'reference signal min: {ppg_ref.min():.4} BPM ({ppg_ref.min()/60:.4} Hz)')
        print(f'reference signal max: {ppg_ref.max():.4} BPM ({ppg_ref.max()/60:.4} Hz)')
        print()
        assert len(ppg)==len(accx)==len(accy)==len(accz), 'PPG and acceleration signal length do not match'
        

def import_input_data(files):
    """
    Import raw input data for the machine learning model
    :param files: list of file names containing the input data 
        (ppg signal, acceleration signals in x-, y- and z-directions)
    :return: dictionary with filename as keys and an inner dictionary as
        values. The inner dictionary contains keys 'ppg', 'accx', 'accy',
        'accz'.
    """
    data = {}
    for file in files:
        name = get_group_name(file_name=file)
        # load data:
        ppg, accx, accy, accz = LoadTroikaDataFile(file)
        data[name] = {
            'ppg': ppg,
            'accx': accx,
            'accy': accy,
            'accz': accz
        }
    return data


def import_labels(files):
    """
    Import labels for the machine learning model.
    :param files: list of file names containing the label data
    :return: dictionary with filename as keys and label data
        (heart rate in BPM) as values
    """
    labels = {}
    for file in files:
        name = get_group_name(file_name=file)
        label = load_labels(file)
        labels[name] = label
    return labels


def window_sizing(dict_signals, window_length, window_shift):
    """
    Split all signals (PPG signal, accelerometer signals, label PPG signals)
    into window sizes given by parameters window_length and window_shift. 
    :param dict_signals: dictionary containing the PPG and accelerometer signals
        labeled 'ppg', 'accx', 'accy', 'accz'.
    :param window_length: length of the window/filter (in sample units)
    :param window_shift: shift of window for each step/ stride (in sample units)
    :return: return dictionary with signal data:  
            'ppg': PPG window data,
            'accx': accelerometer window data in x-direction ,
            'accy': accelerometer window data in y-direction window data,
            'accz': accelerometer window data in z-direction window data,
    """
    data_raw_window = {
        'ppg': [],
        'accx': [],
        'accy': [],
        'accz': [],
    }
    for key, signal in dict_signals.items():
        freqs_total = []
        # loop over window sizes:
        for j in range(0, len(signal), window_shift):
            sig = signal[j:j+window_length]
            # check if we still have a complete window size
            if sig.shape[0]==window_length:
                # add data to dictionary:
                data_raw_window[key] += [sig]

    # transform all dictionary values to pure numpy arrays:
    for key, value in data_raw_window.items():
        data_raw_window[key] = np.stack(value, axis=0)
    return data_raw_window


def match_data_labels(dict_data, labels):
    """
    Reshape data and/or labels so that both have the same length
    (each data point has a label)
    :param dict_data: dictionary with data signals
    :param labels: np.array of labels containing to the data
    :return: data dictionary and np.array of labels with the same
        length
    """
    lst_length = [value.shape[0] for _, value in dict_data.items()] + [labels.shape[0]]
    max_length = min(lst_length)
    ddata = {key: value[:max_length] for key, value in dict_data.items()}
    return ddata, labels[:max_length]


def get_group_name(file_name):
    """
    Get a name for the data (group) generated from the file name
    :param file_name: string containing the filename
    :return: string with group name generated from the file_name
    """
    # for each window append the group (dataset) the signal belongs to:
    return file_name.split('/')[-1][:-4].lower()


def bandpass_filter(signal, pass_band, fs):
    """
    apply a frequency filter (Bandpass filter)
    :param signal: signal in time domain
    :param pass_band: range of frequencies (in Hz) that pass through the filter
    :param fs: sampling frequency
    :return: filtered signal in time domain
    """
    # Design an Nth-order digital or analog Butterworth filter and return the filter coefficients
    # N: order of filter
    # pass_band: critical frequencies
    # fs: sampling frequency of the digital system
    b, a = sp.signal.butter(5, pass_band, btype='bandpass', fs=fs)
    # Apply a digital filter forward and backward to a signal
    return sp.signal.filtfilt(b, a, signal)


def rfft(signal, fs, n_fft, match_shape=False):
    """
    Bandpass a signal and perform a rFFT.
    :param signal: signal to transform from time space to frequency space
    :param fs: sampling frequency
    :param n_fft: FFT window length. If n_fft>len(ppg) the input signal is 
        padded with zeros so that we get a higher resolution in frequency 
        domain
    :param match_shape: boolean that indicates if the output frequencies and
        FFT signal should have the same signal. If True, frequencies np.array
        is repeated until it matches FFT output signal.
    :return: tuple of frequencies and their corresponding rFFT power signal
    """
    # frequencies of fourier transforms
    freqs = np.fft.rfftfreq(n=n_fft, d=1/fs)
    if match_shape:
        freqs = np.tile(np.fft.rfftfreq(n=n_fft, d=1/fs), (signal.shape[0], 1))
    # perform fft:
    # we use n>len(time_series) -> we pad the input signal with
    # zeros so that we get a higher resolution in frequency space
    fft = np.abs(np.fft.rfft(signal, n=n_fft, axis=-1))
    return freqs, fft


def total_accelerometer_signal(accx, accy, accz, n_fft):
    """
    Total accelerometer signal and its FFT. L2-norm of all accelerometer signals.
    :param accx: accelerometer signal in x-direction in time domain
    :param accy: accelerometer signal in y-direction in time domain
    :param accz: accelerometer signal in z-direction in time domain
    :param n_fft: FFT window length. If n_fft>len(ppg) the input signal is 
        padded with zeros so that we get a higher resolution in frequency 
        domain
    :return: tuple of np.array's: (L2-norm of accelerometer signals, rFFT of L2_norm)
    """
    acc = np.sqrt(accx**2 + accy**2 + accz**2)
    # acc_abs = np.sqrt(accx**2 + (accy - np.mean(accy))**2 + accz**2)
    # total accelerometer signal magnitude in frequency domain:
    # variant 1:
    fft_acc = np.abs(np.fft.rfft(acc, n=n_fft, axis=-1))
    return acc, fft_acc


def fft_total_accelerometer_signals(fft_accx, fft_accy, fft_accz):
    """
    Total accelerometer signal based on FFTs of input (directional FFTs).
    All rFFT of accelerometer signals are summed.
    :param fft_accx: rFFT of accelerometer signal in x-direction
    :param fft_accy: rFFT of accelerometer signal in y-direction
    :param fft_accz: rFFT of accelerometer signal in z-direction
    :return: fft_accx + fft_accy + fft_accz
    """
    return fft_accx + fft_accy + fft_accz


def raw_data(input_data_raw, labels_raw, window_length, window_shift, freq_range, fs, n_fft):
    """
    Take the input data and labels and transform them to the raw data split into chunks (window
    sizes). The output raw data serves as basis for calculating the features for the machine
    learning model.
    :param input_data_raw: dictionary with filename as keys and an inner dictionary as
        values. The inner dictionary contains keys 'ppg', 'accx', 'accy',
        'accz'.
    :param labels_raw: dictionary with filename as keys and label data
        (heart rate in BPM) as values
    :param window_length: length of the window/filter (in sample units)
    :param window_shift: shift of window for each step/ stride (in sample units)
    :param freq_range: tuple with minimum and maximum frequency.
        The frequencies and fft signals are filterd for frequencies in this interval range.
    :param fs: sampling frequency
    :param n_fft: FFT window length. If n_fft>len(ppg) the input signal is 
        padded with zeros so that we get a higher resolution in frequency 
        domain
    :return: tuple with 3 elements:
        dict_data_raw_window: all raw datapoints (signals split into fixed window sizes)
        labels: np.array with labels for all raw data data points (heart rate in BPM)
        groups: group each datapoint belongs to. As group name the file name of each
            datapoint is taken.

    """
    if labels_raw:
        labels = np.array([])
        groups = np.array([])
        lbls = {}
    else:
        labels = None
        groups = None
    dict_data_raw_window = {
        'ppg': [],
        'accx': [],
        'accy': [],
        'accz': [],
    }
    for key, dict_sig in input_data_raw.items():
        dict_data_window = window_sizing(
            dict_signals=dict_sig,
            window_length=window_length,
            window_shift=window_shift,
        )
        if labels_raw:
            # bring/ensure input data and labels to/have same length:
            dict_data_window, lbls[key] = match_data_labels(
                dict_data=dict_data_window,
                labels=labels_raw[f'ref{key[4:]}']
            )
            labels = np.append(arr=labels, values=lbls[key], axis=0)
            # add a group label so that one can assign each (sub-)
            # data to a dataset:
            grps = np.array([key]*len(lbls[key]))
            groups = np.append(arr=groups, values=grps, axis=0)
        for k, val in dict_data_window.items():
            dict_data_raw_window[k] += [val]

    # concatenate all inner dicitonaries:
    for key, val in dict_data_raw_window.items():
        dict_data_raw_window[key] = np.row_stack(val)

    # calculate remaining relevant signals:
    dict_temp = {}
    for key, val in dict_data_raw_window.items():
        sig_filtered = bandpass_filter(
            signal=val,
            pass_band=freq_range,
            fs=fs
        )
        freqs, fft = rfft(signal=sig_filtered, fs=fs, n_fft=n_fft, match_shape=True)

        dict_temp[f'{key}_filtered'] = sig_filtered
        dict_temp['freqs'] = freqs
        dict_temp[f'fft_{key}'] = fft
    # concatenate dictionaries:
    dict_data_raw_window = {**dict_data_raw_window, **dict_temp}

    # total accelerometer signal 'abs':
    dict_data_raw_window['acc_abs_filtered'], dict_data_raw_window['fft_acc_abs'] = total_accelerometer_signal(
        accx=dict_data_raw_window['accx_filtered'],
        accy=dict_data_raw_window['accy_filtered'],
        accz=dict_data_raw_window['accz_filtered'],
        n_fft=n_fft
    )
    # total accelerometer signal 'sum':
    dict_data_raw_window['fft_acc_sum'] = fft_total_accelerometer_signals(
        fft_accx=dict_data_raw_window['fft_accx'],
        fft_accy=dict_data_raw_window['fft_accy'],
        fft_accz=dict_data_raw_window['fft_accz']
    )
    return dict_data_raw_window, labels, groups


def reshape_peaks(peaks, freqs, fft, nr_peaks=4, pad_value=np.nan):
    """
    Reshape the calculated peaks in FFT. The peaks are first
    sorted by their power spectrum
    :param peaks: array of indices with peaks of the fft signal
    :param freqs: frequencies corresponding to the fft signal
    :param fft: fft power spectrum signal the peaks are extracted from
    :param nr_peaks: number of peaks we want to output. If there are
        less peaks detected than nr_peaks requires then the array is 
        padded with value pad_value until the length of the returned 
        arrays are equal to nr_peaks otherwise the array is cut after
        the nr_peaks are reached.
    :param pad_value: value to use for padding
    :return: tuple of 2 numpy.array: (frequency of peaks, y-value of peak)
    """
    # get the sorted peak indices:
    # we sort by the frequency power spectrum (fft) from hight to low:
    peaks_sorted = peaks[fft[peaks].argsort()[::-1]]

    # get the sorted fft power spectrum:
    pwr_peaks_sorted = fft[peaks_sorted]
    # pad all peaks to a certain length so that all ffts have the same length
    pwr_peaks_sorted = np.pad(
        array=pwr_peaks_sorted[:nr_peaks],
        pad_width=(0, nr_peaks-len(pwr_peaks_sorted[:nr_peaks])),
        constant_values=pad_value
    )

    # get the sorted frequencies (first frequency is assigned to 
    # the most promintent peak in the frequency power spectrum, and so on):
    freqs_peaks_sorted = freqs[peaks_sorted]
    freqs_peaks_sorted = np.pad(
        array=freqs_peaks_sorted[:nr_peaks],
        pad_width=(0, nr_peaks-len(freqs_peaks_sorted[:nr_peaks])),
        constant_values=pad_value
    )
    return freqs_peaks_sorted, pwr_peaks_sorted


def find_transform_peaks(
    freqs,
    fft,
    nr_peaks=4,
    pad_value=np.nan,
    pk_height=None,
    pk_distance=None,
    pk_prominence=None,
    pk_wlen=None
):
    """
    Find peaks of a FFT signal and transform their output so that each
    signal has the same length (e.g. return the first 4 peaks if nr_peaks=4).
    Peaks are identified based on the peak parameters pk_height, pk_distance,
    pk_prominence and pk_wlen.
    :param freqs: frequencies corresponding to the fft signal
    :param fft: fft power spectrum signal the peaks are extracted from
    :param nr_peaks: number of peaks we want to output. If there are
        less peaks detected than nr_peaks requires then the array is 
        padded with value pad_value until the length of the returned 
        arrays are equal to nr_peaks otherwise the array is cut after
        the nr_peaks are reached.
    :param pad_value: value to use for padding
    :param pk_height: Required height of peaks. Since the signal is normed in range
        [0, 1] pk_height should be in range [0, 1]
    :param pk_distance: Required minimal horizontal distance (>= 1) in samples between 
        neighbouring peaks. Smaller peaks are removed first until the condition is
        fulfilled for all remaining peaks.
    :param pk_prominence: The peak prominence measures how much a peak stands out
        from the surrounding baseline of the signal and is defined as the vertical distance
        between the peak and its lowest contour line. Since the signal is normed in range
        [0, 1] pk_prominence should be in range [0, 1]
    :param pk_wlen: Used for calculation of the peaks prominences. A window length in samples
        that optionally limits the evaluated area for each peak to a subset of x. The peak is
        always placed in the middle of the window therefore the given length is rounded
        up to the next odd integer
    :return: tuple of 2 numpy.array: (frequency of peaks, y-value of peak)
    """
    peaks_freqs = []
    peaks_pwr = []
    for fqs, ft in zip(freqs, fft):
        # norm signal:
        # we norm each signal independently to the range [0, 1] so that we can find the peaks more easily.
        min_new = 0
        max_new = 1
        ft_normed = (max_new - min_new) / (ft.max() - ft.min()) * (ft - ft.max()) + max_new
        ## calculate frequency peaks in all relevant signals and relevant frequency range:
        ## prominence of a peak: measures how much a peak stands out from the surrounding
        ##     baseline of the signal and is defined as the vertical distance between the peak
        ##     and its lowest contour line.
        ## threshold of peaks: the vertical distance to its neighboring samples.
        ## wlen: Used for calculation of the peaks prominences. A window length in samples that 
        ##     optionally limits the evaluated area for each peak to a subset of x. The peak is
        ##     always placed in the middle of the window therefore the given length is rounded
        ##     up to the next odd integer
        peaks, properties = sp.signal.find_peaks(
            x=ft_normed,
            height=pk_height if pk_height is not None else None,
            distance=pk_distance,
            prominence=pk_prominence if pk_prominence is not None else None,
            threshold=None,
            wlen=pk_wlen
        )
        freqs_peaks, pwr_peaks = reshape_peaks(
            peaks=peaks,
            freqs=fqs,
            fft=ft,
            nr_peaks=nr_peaks,
            pad_value=pad_value
        )
        peaks_freqs += [freqs_peaks]
        peaks_pwr += [pwr_peaks]
    return np.stack(peaks_freqs, axis=0), np.stack(peaks_pwr, axis=0)


def fractional_spectral_energy(freqs, fft, freq_range):
    """
    Fractional spectral energy of fourier power signal.
    :param freqs: frequencies for corresponding fourier signal
	(same unit as freq_range)
    :param fft: power spectrum of fourier signal
    :param freq_range: tuple with minimum and maximum frequency.
        The minimal and maximal frequency should be contained in the
        freqs array range (same unit as freqs)
    :return: fractional energy
    """
    # if inputs are not 2d transform them to 2d
    fft = np.atleast_2d(fft)
    freqs = np.atleast_2d(freqs)
    # spectral energy of complete signal (without summation)
    spectral_energy = np.square(np.abs(fft))
    # mask frequency range that is considered for comparison to complete signal:
    msk = (freqs>freq_range[0]) & (freqs<=freq_range[1])
    # divide interesting
    return (spectral_energy * msk).sum(axis=1) / np.sum(spectral_energy, axis=1)


def filter_frequencies(freqs, fft, freq_range):
    """
    Filter a FFT signal by a frequency range.
    :param freqs: frequencies corresponding to the fft signal
    :param fft: fft power spectrum signal
    :param freq_range: tuple with minimum and maximum frequency.
        The frequencies and fft signals are filterd for frequencies in this interval range.
    :return: tuple of numpy arrays:
        - filtered frequencies: frequencies that lie inside interval [min_freq, max_freq] and
            correspond to the fft signal. All frequencies outside of this interval are set to
            zero
        - filtered FFT signal: FFT signals that belong to the frequencies. All FFT signals that
            correspond to frequencies outside of the interval [min_freq, max_freq] are set to
            zero
    """
    # create mask for selecting frequencies that lie in frequency
    # range of interest [min_freq, max_freq]:
    msk = (freqs>=freq_range[0]) & (freqs<=freq_range[1])
    freqs_filtered = freqs * msk
    fft_filtered = fft * msk
    return freqs_filtered, fft_filtered


def featurize(dict_raw_window_data, peaks_param, frac_en_param, freq_range):
    """
    Calculate all features from the raw input signals.
    :param dict_raw_window_data: dictionary with raw time series (appropriate window size)
        of all relevant signals
    :param peaks_param: dictionary with peak parameters for each signal the peak features
        should be calculated
    :param frac_en_param: dictionary with fractional energy parameter for each siagnal the
        fractional energies should be calculated
    :param freq_range: tuple with minimum and maximum frequency.
        The frequencies and fft signals are filterd for frequencies in this interval range.
    :return: dictionary with all specified features
    """
    # initialize dictionary containing all features
    features = {}
    
    # iterate over each raw signal available:
    for str_short in {*peaks_param.keys(), *frac_en_param.keys()}:
        # filter fft signal for relevant frequencies
        # (all frequencies outside of interval freq_range
        # are filtered out so that no peak is found outside
        # the frequency interval of interest):
        freqs_filtered, fft_filtered = filter_frequencies(
            freqs=dict_raw_window_data['freqs'],
            fft=dict_raw_window_data[f'fft_{str_short}'],
            freq_range=freq_range
        )
        
        if str_short in peaks_param.keys():
            # find peaks:
            features[f'peaks_freq_{str_short}'], features[f'peaks_pwr_{str_short}'] = find_transform_peaks(
                freqs=freqs_filtered,
                fft=fft_filtered,
                nr_peaks=peaks_param[str_short]['nr_peaks'], 
                pad_value=peaks_param[str_short]['pad_value'],
                pk_height=peaks_param[str_short]['height'],
                pk_distance=peaks_param[str_short]['distance'],
                pk_prominence=peaks_param[str_short]['prominence'],
                pk_wlen=peaks_param[str_short]['wlen']
            )
        if str_short in frac_en_param.keys():
            # get frequency intervals
            frequency_intervals = np.linspace(
                start=freq_range[0],
                stop=freq_range[1],
                num=frac_en_param[str_short]['num_intervals'],
                endpoint=True,
                retstep=False,
                dtype=None,
                axis=0
            )
            # iterate over all frequency intervals:
            for i, freq_r in enumerate(zip(frequency_intervals[:-1], np.roll(frequency_intervals, -1)[:-1])):
                # calculate fractional spectral energy in the given frequency interval:
                features[f'frac_en_int{i}_{str_short}'] = fractional_spectral_energy(
                    freqs=freqs_filtered,
                    fft=fft_filtered,
                    freq_range=freq_r
                )
    return features
