# Function Bank
import pickle
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Shortcuts
'Ctrl + Alt + Shift + [ = Collapse all functions'


def t_check(starts, times, num_trials, num_seq, num_emoji):
    # Converts 1-D array of all flash event marker ground truths into numeric and reshapes to Trials x Sequences x Emoji.

    starts = np.reshape(starts, (num_trials, num_seq, num_emoji))
    print(starts - times[0])

    return starts


def mark_stamps(markers_file, num_trials, num_seq, num_emoji):
    # Converts 1-D array of all flash event marker ground truths into numeric and reshapes to Trials x Sequences x Emoji.

    starts = np.reshape(np.load(markers_file)['arr_1'], (num_trials, num_seq, num_emoji), order='C')
    ends = np.reshape(np.load(markers_file)['arr_3'], (num_trials, num_seq, num_emoji), order='C')

    return starts, ends


def mark_arrang(markers_file, num_trials, num_seq, num_emoji):
    # Converts 1-D array of all flash event marker ground truths into numeric and reshapes to Trials x Sequences x Emoji.

    mark = np.load(markers_file)['arr_0']
    mark = ','.join(mark)

    new = []
    for i in range(len(mark)):
        new = np.append(new, np.fromstring(mark[i], dtype=int, sep=','))

    new = np.reshape(new.astype(int), (num_trials, num_seq, num_emoji))
    return mark


def plot_pro(fig_num, title, data, x_axis, leg, show, verbose):
    '''
    Simple quick plotting function for cleaner code in main functions.

    Assumes Samples x Channels / Events / Sequences / Trials.

    Inputs:

    fig_num     = figure number.
    title       = figure title.
    data        = time-series data (e.g. mV amplitudes/ impedance ohms).
    x_axis      = temporal data of EEG times-series.
    leg         = plot legend (e.g. channel indices ['Fz', 'Cz', 'A2'])
    show        = call plt.show() to publish figures.
    verbose     = print function info.

    Example:

    plot_pro(1, 'Raw Zeroed Data', eeg, None, leg=chan_ind, show=None, verbose=1)

    '''
    # Get data dims.
    a, b = np.shape(data)
    # Assign figure number.
    plt.figure(fig_num)
    # Assign figure title.
    plt.title(title)
    # If no x_axis generate one based on len of samples.
    if x_axis is None:
        x_axis = np.arange(a)
    '---Plot Data'
    plt.plot(x_axis, data)
    # If legend specified, assign.
    if leg is not None:
        plt.legend(leg, loc='upper right', fontsize='xx-small')
    plt.tight_layout()
    # Print data info.
    if verbose == 1:
        print('EEG DIMS: ', data.shape)
    if show == 1:
        plt.show()


def filtering(eeg, samp_ratekHz, zero, hlevel, llevel, notc, notfq):
    '''
    Applies filtering to EEG time-series data.

    Assumes Samples x Channels.

    Inputs:

    eeg             = data matrix.
    samp_ratekHz    = data acquisition sampling rate in kHz e.g. 0.5 == 500Hz.
    zero            = if 'ON' data will be zeroed by subtracting the mean of the channel array.
    hlevel          = high-pass filter limit, if set at None no high-pass filtering is applied.
    llevel          = low-pass filter limit, if set at None no low-pass filtering is applied.
    notc            = notc filter options for removing 50Hz noise, either 'NIK', 'NOC' or 'LOW'
                      see each respective function for more details.
    notfq           = frequency at which to apply notch filter, UK powerline is 50Hz.


    Outputs:

    data            = filtered data.

    Example: data = pb.filtering(data, samp_ratekHz=0.5, zero='ON', hlevel=0.1, llevel=30, notc='NIK', notfq=50)

    '''
    #
    '---------------------------------------------------'
    'ZERO: mornalize data with zero meaning.'
    if zero == 'ON':
        eeg = zero_mean(np.copy(eeg))
    '---------------------------------------------------'
    'FILTERING: highpass filter.'
    if hlevel is not None:
        eeg = butter_highpass_filter(np.copy(eeg), hlevel, 500, order=5)
    '---------------------------------------------------'
    'FILTERING: lowpass filter.'
    if llevel is not None:
        # Fix by adding singleton dimension.
        if len(np.shape(eeg)) == 1:
            eeg = np.expand_dims(np.copy(eeg), axis=1)
        eeg = butter_lowpass_filter(np.copy(eeg), llevel, 500, order=5)
    '----------------------------------------------------'
    'FILTERING: 50Hz notch filter.'
    if notc == 'NIK':
        samp_rateHz = samp_ratekHz * 1000
        eeg = notchyNik(eeg, Fs=samp_rateHz, freq=notfq)
    elif notc == 'NOC':
        eeg = notchy(eeg, 500, notfq)
    elif notc == 'LOW':
        eeg = butter_lowpass_filter(eeg, notfq, 500, order=5)
    return eeg


def range_remov(data, n_chans, chan_ind, limit, verbose):
    '''
    Removes channels if mV amplitudes exceed certain limitand is especially useful for plotting.

    Inputs:

    data        = eeg time-series.
    n_chans     = number of eeg channels.
    chan_ind    = string list of channel names included e.g. ['Fz', 'Cz', 'A2']
    limit       = Limit acts like a band. If set at 40, any channel containing values greater than 40
                  or lower than -40 will be marked and excluded.
    verbose     = prints function info.

    Outputs:

    data        = data with channels exceeding limit removed.
    chan_list   = string list of channels retained after limit checking.

    NOTE: If all channels contain signals exceeding

    Assumes Samples x Channels.

    Example:

    data, chan_ind =  pb.range_remov(data, n_chans, chan_ind=chan_ind, limit=40, verbose=0)

    '''

    # Identify and Mark Channels with values.
    x_list = np.arange(n_chans)
    for q in range(n_chans):
        if np.any(data[:, q] > limit) is True or np.any(data[:, q] < np.negative(limit)) is True:
            print('Offending Channel: ', chan_ind[q])
            x_list[q] = 2000

    # Amend List to contain indices of high quality channels.
    fin_list = []
    chan_list = []
    for r in range(n_chans):
        if x_list[r] != 2000:
            fin_list = np.append(fin_list, x_list[r])
            chan_list = np.append(chan_list, chan_ind[r])
    fin_list = fin_list.astype(int)

    # Slice high quality channels from low quality channels.
    data = data[:, fin_list]

    if verbose == 1:
        print('Channels Retained from Range Removal: ', chan_list)
        print('RANGE REMOV DATA DIMS: ', data.shape)

    return data, chan_list


def referencing(data, chan_ind, type, limit, verbose):
    '''
    Referencing in either A2 subtraction or average referencing with range removal.

    Inputs:

    data        = eeg time-series.
    chan_ind    = string list of channel names included e.g. ['Fz', 'Cz', 'A2']
    type        = 'A2' or 'AVG'.
    limit       = 'AVG' referencing makes use of the range_removal function, this acts like a band boolean.
                  If set at 40, any channel containing values greater than 40 or lower than -40
                  will be marked and excluded.
    verboe      = print function info.

    Outputs:

    data      = 'AVG' refereced data

    NOTE: Assumes data in Samples x Channels.
    NOTE: Assumes A2 channel is in data matrix.
    NOTE: channel used for referencing in A2 style must be position as last channel (-1) in matrix.

    Example: data = pb.referencing(data, type='AVG', limit=20)

    '''

    # Data Dimensions.
    samp, n_chans = np.shape(data)
    # Isolate Reference Channel
    ref_chan = data[:, -1]
    # Remove Reference Channel from data matrix.
    data = data[:, 0:-1]

    if type == 'A2':
        for i in range(n_chans - 1):
            data[:, i] = data[:, i] - ref_chan
        if verbose == 1:
            # Data Dimesions Check:
            print('Post A2 REF EEG DIMS: ', data.shape)
        return data

    if type == 'AVG':
        # Use range_removal function to remove channels with excessive range_values.
        data, chan_list = range_remov(data, n_chans=n_chans - 1,
                                      chan_ind=chan_ind, limit=limit, verbose=1)
        # Generate average reference from eeg montage.
        ref_avg = np.average(data, axis=1)
        # Subtract average reference from each eeg channel.
        for i in range(n_chans - 1):
            data[:, i] = data[:, i] - ref_avg
        if verbose == 1:
            # Data Dimesions Check:
            print('Post AVG REF EEG DIMS: ', data.shape)
        return data


def reducer(data, type, ref, chan_list, verbose):
    '''
    Signal Reduction to Samples x Emoji / Event matrix.
    Reduce array down to just one signal either by isolating Cz or averaging across all OR a number of channels.

    Assumes Samples x Channels x Emoji.

    Inputs:

    data        = eeg time-series.
    type        = either 'AVG' meaning an average signal is generated based off of the channel indices in chan_list,
                  or 'IND', meaning a single channel e.g. Cz  is simply taken from the list and added to a separate
                  output matrix.
    ref         = matrix channel index for electrode you want to individually ('IND') isolate, if None, then skip.
    chan_list   = list of channel indices for cross-channel averaging.
    verbose     = print function info.

    Outputs:

    data        = single channels of emoji level data (Samples x Channels x Emoji Events).
                  Channel dimension is just a singleton dimensions to retain features of following steps.

    IND Example: i_data = reducer(r_data, type='IND', ref=[1], chan_list=None, verbose=1)
    AVG Example: i_data = reducer(r_data, type='AVG', ref=None, chan_list=[0, 1], verbose=1)


    '''

    if type == 'IND':
        data = data[:, ref, :]
    elif type == 'AVG':
        data = np.average(data[:, chan_list, :], axis=1)
        data = np.expand_dims(data, axis=1)
    if verbose == 1:
        print('Post-Reduction DATA DIMS: ', data.shape)

    return data


def pipeline(num_chan, num_emoji, num_seq, starts, ends, cue, order, eeg, eeg_time, seq_c, chan_ind, detrend, out_size, main_rs, plots, verbose=1):
    '''
    Pre-process 1 sequence worth of data.

    Inputs:

    num_chan    = number of EEG channels in input array.
    num_emoji   = number of emoji in stimuli display.
    num_seq     = number of sequences per trial.
    starts      = list of time_stamps relating to the initiation of the sequence.
    seq_s2      = list of labels presented at the end of the emoji event.
    ends        = list of time_stamps relating to the end of the sequence.
    cue         = emoji cued throughout the sequence.
    order       = order of emoji flashing over the sequence period.
    eeg         = eeg data collected during the sequence.
    eeg_time    = timestamps for the eeg data collected.
    seq_c       = count for sequence currently being processed.
    detrend     = application of polynomial detrending of order 10 (1 == Yes).
    out_size    = number of samples (corresponds to seconds of data captured) you want in the output matrix.
    main_rs     = matrix reshaping for main experiment.
    plots       = plot movement through the pipeline.
    verbose     = info printing (1 == yes).

    '''

    '====Sequence Level Pre-Pro===='
    '----Zeroing----'
    eeg = zero_mean(eeg)
    if plots == 1:
        plot_pro(1, 'Raw Zeroed Data', eeg, eeg_time, leg=chan_ind, show=None, verbose=0)
    '----Detrending----'
    if detrend == 1:
        eeg = pol_det(np.copy(eeg), chan_ind, order=10, plot=False)
    if plots == 1:
        plot_pro(2, 'Post-Detrend Data', eeg, eeg_time, leg=chan_ind, show=None, verbose=0)
    '----Referencing----'
    eeg = referencing(eeg, chan_ind=chan_ind, type='A2', limit=15, verbose=0)
    if plots == 1:
        plot_pro(4, 'Post-Referencing Data', eeg, eeg_time, leg=chan_ind, show=1, verbose=0)

    '----Preform Matrices----'
    # Aggregate arrays: Samples x Channels x Emoji x Sequences
    'NOTE: num_chan - 1, as the A2 electrode has been removed via referencing function.'
    r_data = np.zeros((out_size, num_chan - 1, num_emoji))
    # Aggregate arrays for NON-zeroed timestamps plotting.
    r_times = np.zeros((out_size, num_emoji))
    # Aggregate arrays for zeroed timestamps plotting.
    r_times_zer = np.zeros((out_size, num_emoji))
    # Post-moji event buffer lengeth calculation (0.5 being the sample rate in kHz).
    buffer = (out_size / 1000) / 0.5

    for e in range(num_emoji):
        '----Segmentation----'
        # START: Find nearest value of the marker timestamps in the corresponding data timestamp array.
        v_s = starts[seq_c, e]
        # Index in timestamp array closest to onset of marker indcating the start of the emoji event.
        str_idx = (np.abs(eeg_time - v_s)).argmin()
        # Pad to start marker + buffer to ensure all P3 wave form extracted and indexing to that location in the data array.
        if ends[seq_c, e] < starts[seq_c, e] + 0.3:
            v_e = starts[seq_c, e] + buffer
        else:
            # Just a check to ensure the end marker is not below 0.3s (past peak of the P3 waveform).
            print('Crash Code: End Marker Positioned Before P3 Propogation.')
            v_e = starts[seq_c, e] + buffer
        # Index in timestamp array closest to onset of marker indcating the end of the emoji event.
        end_idx = (np.abs(eeg_time[:] - v_e)).argmin()
        # Indexing into data array to extract currect P300 chunk.
        seq_chk = eeg[str_idx: end_idx, :]
        'Timing variables'
        # Non-Zeroed Timestamps @ Sequence Level.
        seq_temp = eeg_time[str_idx: end_idx]
        # Zeroed Timestamps @ Sequence Level.
        seq_temp_zer = seq_temp - seq_temp[0]
        if plots == 1:
            plot_pro(5, 'Post-Segment Data', seq_chk, seq_temp_zer,
                     leg=chan_ind, show=None, verbose=0)
        '----Resample----'
        # Resampling Interpolation Method @ Channel Level, using zeroed timestamp values.
        r_data[:, :, e], r_times_zer[:, e] = interp2D(
            seq_chk, seq_temp_zer, output_size=out_size, plotter=0, verbose=0)
        if plots == 1:
            plot_pro(6, 'Post-Resample Data', r_data[:, :, e],
                     r_times_zer[:, e], leg=chan_ind, show=None, verbose=0)
        '----Pre-Processing----'
        # 0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.
        r_data[:, :, e] = filtering(r_data[:, :, e], samp_ratekHz=0.5, zero='ON', hlevel=0.1,
                                    llevel=12, notc='NIK', notfq=50)
        if plots == 1:
            plot_pro(3, 'Post-Pre-Pro Data', eeg, eeg_time, leg=chan_ind, show=None, verbose=0)
    '----Signal Reduction----'
    # Reduce array down to just one signal either by isolating Cz or averaging across all OR a number of channels.
    # Output should be 2D Samples x Emoji.
    # i_data = reducer(r_data, type='IND', ref=[1], chan_list=None, verbose=0)
    i_data = reducer(r_data, type='AVG', ref=None, chan_list=[0, 1, 2, 3, 4, 5, 6], verbose=1)
    if plots == 1:
        plot_pro(7, 'Post-Channel Reduction Data', i_data[:, :, e],
                 r_times_zer[:, e], leg=None, show=None, verbose=0)
    'Main EXP Reshape:'
    if main_rs == 1:
        i_data = np.squeeze(i_data)
    # i_data = np.expand_dims(i_data, axis=2)
    '----Info-----'
    if verbose == 1:
        print('====EMOJI {}===='.format(e + 1))
        print('Start / V_s: ', v_s, 'End / V_E: ', v_e)
        print('Str Idx: ', str_idx, 'End Idx: ', end_idx)
        print('R_TIMES DIMS: ', r_times.shape)
        print('R_TIMES_ZER DIMS: ', r_times_zer.shape)
        print('R_DATA (Resampled) DIMS: ', r_data.shape)
        print('I_DATA (ISOLATED / REDUCED) DIMS', i_data.shape)
    '----Return Aggregate Matrices.'
    return i_data, r_times_zer
    '''
    NOTE: , bin_labels, n_bin_labels
    '''


def sig_avg(data, out_size, num_emoji, num_seq, order, scaler, plots, verbose):
    '''
    Averages signals for each emoji location across sequences for randomised event timings based off event marker timestamps.
    Irrespective of how many stimuli you have this method should work, with emoji pos 1 being 1st in the matrix.

    Inputs:

    dast        = input eeg time-series.
    out_size    = number of samples in output matrix.
    num_emoji   = number of emoji in stimuli array.
    num_seq     = number of sequences per experimental trial.
    order       = temporal augmentation order of emoji (indices represents spatial positioning left-to-right).
    scaler      = normalization method post-averaging either 'min' (minMax method) or 'standard' (standard scikitLean scaler).
    plots       = switch for plotting data post-normalization.
    verbose     = print function info.

    Outputs:

    data        = averaged and possibly normalized data.

    NOTE: This averages across emoji events ACCOMODATING for the variable flash time, therefore output is ordered
          spatially and able to simply connect the emoji spatial location with the target cue index.
    '''
    # Preform for averages across entire trial.
    a_data = np.zeros((out_size, num_emoji))
    # Pre-form for single emoji worth of data across entire trial.
    e_data = np.zeros((out_size, num_seq))

    # Averaging across time for each segmented emoji location.
    for e2 in range(num_emoji):
        for s2 in range(num_seq):
            e_data[:, s2] = data[:, order[e2, s2], s2]
        a_data[:, e2] = np.average(e_data, axis=1)
        '----Normalization-----'
        # Feature Scaling.
        if scaler == 'min':
            a_data[:, e2] = min_max_scaler(a_data[:, e2], fr=(-5, 5))
        elif scaler == 'standard':
            a_data[:, e2] = stand_scaler(a_data[:, e2])
        if plots == 1:
            plot_pro(8, 'Post-Normalization Data',
                     np.expand_dims(a_data[:, e2], axis=1), x_axis=None, leg=None, show=1, verbose=0)

    if verbose == 1:
        print('INPUT DATA DIMS: ', data.shape)
        print('EMOJI DATA DIMS: ', e_data.shape)
        print('AVG DATA DIMS: ', a_data.shape)
        print('Sequence Augmentation Order: \n', (order))

    return a_data


def model_app(data, type, cue, model):
    '''
    type        = either offline ('OFFLINE') just returning data for later classifier calibration
                  or online ('ONLINE') using a pre-calibrated model for prediction & returning data.
    mod_file    = location of the lda model used for real-time predicition.
    '''
    if type == 'OFFLINE':
        print('REQUIRES DEV.')
    elif type == 'ONLINE':
        '====REAL_TIME PREDICTION===='
        # Real-Time Prediction @ emoji event level.
        predictions = model.predict(data)
        pred_proba = model.predict_proba(data)
        print('------------------------------')
        print('------------RESULT------------')
        print('Targ Cue: ', cue)
        print('Predictions:   ', predictions.shape, predictions)
        # p3_pick.
        print('Predict Probabilities: \n', np.round(pred_proba, decimals=2))
        if cue == np.argmax(pred_proba[:, 1]):
            binary_val = 1
            print('CORRECT RETURN | TARGET CUE: ', cue, '| NON-BINARY ANSWER: ', np.argmax(
                pred_proba[:, 1]), '| BINARY: ', binary_val, '| PROBABILITY: ', np.amax(pred_proba[:, 1]))
            # result = Binary Prediction / targ_cue = Cued Emoji / pred_em = prediction from emoji locations / proba = Probability.
            return binary_val, np.argmax(pred_proba[:, 1]), np.amax(pred_proba[:, 1])
        else:
            binary_val = 0
            print('INCORRECT RETURN | TARGET CUE: ', cue, '| NON-BINARY ANSWER: ', np.argmax(
                pred_proba[:, 1]), '| BINARY ANSWER: ', binary_val, '| PROBABILITY: ', np.amax(pred_proba[:, 1]))
            # result = Binary Prediction / targ_cue = Cued Emoji / pred_em = prediction from emoji locations / proba = Probability.
            return binary_val, np.argmax(pred_proba[:, 1]), np.amax(pred_proba[:, 1])


def stat_check(data, chan_ind, verbose):
    '''
    Prints Basic Stats.

    Assumes Samples x Channels.

    '''

    for w in range(len(chan_ind)):
        imean = np.mean(data[:, w])
        imin = np.amin(data[:, w])
        imax = np.amax(data[:, w])
        irange = imax - imin
        # Info
        print('---Channel-Wise Stats: ', chan_ind[w])
        print(chan_ind[w], ' Mean: ', imean)
        print(chan_ind[w], ' Min: ', imin)
        print(chan_ind[w], ' Max: ', imax)
        print(chan_ind[w], ' Range: ', irange)


def stand_scaler(X):
    '''
    Standard Scaler from SciKit Learn package.

    Assumes 1D Samples array.

    Input:

    X       = eeg time-series data.
    fr      = feature range e.g. (-5, 5)

    Output:

    X       = standardized data.

    Example:    stand_X = pb.stand_scaler(X)

    '''
    from sklearn.preprocessing import StandardScaler
    # https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
    sc = StandardScaler()
    # Reshape pre-normalization.
    X = X.reshape(-1, 1)
    X = sc.fit_transform(X)
    X = np.squeeze(X)

    return X


def min_max_scaler(X, fr):
    '''
    Min Max Scaler from SciKit Learn package.

    Assumes 1D Samples array.

    Input:

    X       = eeg time-series data.
    fr      = feature range e.g. (-5, 5)

    Output:

    X       = standardized data.

    Example:    mms_X = pb.min_max_scaler(X, fr)

    '''
    from sklearn.preprocessing import MinMaxScaler
    # https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
    mss = MinMaxScaler(feature_range=fr)
    # Reshape pre-normalization.
    X = X.reshape(-1, 1)
    X = mss.fit_transform(X)
    X = np.squeeze(X)

    return X


def lin_det(data, chan_ind, plot):
    '''
    Applies Simplistic Linear Detrending to EEG time-series.

    Assumes Samples x Channels.

    References:
    https://machinelearningmastery.com/time-series-trends-in-python/

    '''
    # Linear Detrending.
    from scipy.signal import detrend

    for b in range(len(chan_ind)):
        if plot == True:
            plt.subplot(2, 1, 1)
            plt.title(chan_ind[b])
            plt.plot(data[:, b])
        # Linear Detrending.
        data[:, b] = detrend(data[:, b])
        if plot == True:
            plt.subplot(2, 1, 2)
            plt.plot(data[:, b])
            plt.show()
    return data


def pol_det(data, chan_ind, order, plot):
    '''
    Calculate Polynomial Line of Fit to Data and Detrend.

    Inputs:

    data        = eeg time-series.
    chan_ind    = list of string relating to eeg channels included e.g. ['Fz', 'Cz', 'A2']
    order       = fidelity of the polynomial line of fit.
    plots       = use built-in obspy detrend plotter (True vs False)

    Output:

    data       = detrended time-series.

    Example: data = pol_det(data, chan_ind, order=10, plot=True)

    Assumes Samples x Channels.

    References:

    https://machinelearningmastery.com/time-series-trends-in-python/
    https://docs.obspy.org/packages/autogen/obspy.signal.detrend.polynomial.html

    ObsPy Install Guide:

    https://github.com/obspy/obspy/wiki/Installation-on-Windows-from-Source

    '''

    # Polynomial Data Detrending.
    import obspy
    from obspy.signal.detrend import polynomial
    for t in range(len(chan_ind)):
        polynomial(data[:, t], order=order, plot=plot)
    return data


def class_balance(X, y, balance, verbose):
    '''

    Generates a class balance of P3 and Non-P3 events based on Binary class labels.

    1) grabs all indices of the P3 and NP3 events from the string labels array and stores separately.
    2) shuffles the NP3 label indices array to ensure sampling across entire session.
    3) cuts down NP3 label indices array to same size as that of the P3 labels indices array for
        a clean 50/50 split to maintain class blance in analysis model fits and testing.
    4) aggregates P3 and non-P3 data according to these indices into separate data arrays.
    5) brings all P3 and non-P3 data and labels together for Striatified Shuffling or randomized
        subsampling further down the analysis pipeline.

    NOTE: Assumes Events x Samples.

    Inputs:

    data = aggregated segmented EEG data across session.
    labels = MUST BE BINARY ground truth string labels indicating eeg event P3 / NP3.
    balance = refers to a percentage difference in terms of P3 vs NP3 events in final aggregate array.
                For example, a value of 1 means 1:1 ratio, a value of 2 means a 1:2 / P3:NP3 ratio.
                This has a hard-limit as there are only so many P3s to NP3, recommend max of 5.
    verbose = info on process 1 == print.

    Outputs:

    bal_data = aggregated class-balanced data matrix.
    bal_labels = aggregated class-balanced string labels array.
    bal_i_labels = aggregated class-balanced numeric labels array.

    Example:

    bal_i_labels, bal_data, bal_labels = pb.class_balance_50_50(X, y)

    '''
    import random
    # Preform P3 and NP3 agregate arrays.
    p3 = []
    np3 = []
    # Gather all indices of P3 (1) and NP3 (0) events in the labels array.
    id_np3 = [i for i, x in enumerate(y) if x == 0]
    id_p3 = [i for i, x in enumerate(y) if x == 1]
    # Convert to subscriptable numpy array.
    id_np3 = np.asarray(id_np3)
    id_p3 = np.asarray(id_p3)
    # Convert to numeric tye integer for indexing.
    id_np3 = id_np3.astype(int)
    id_p3 = id_p3.astype(int)
    # Shuffle the Non-P300 event indices list.
    random.shuffle(id_np3)
    # Balance Value.
    bal_val = np.int(len(id_p3) * balance)
    # Reduce the number of NP3 indices to the amount examples secified vai the balance value.
    id_np3 = id_np3[0:bal_val]
    # Print function info.
    if verbose == 1:
        print('ID_P3 / NUMBER OF P3 EVENTS: ', len(id_p3))
        print('ID_NP3 / NUMBER OF NP3 EVENTS PRE SHUFFLE: ', len(id_np3))
        print(id_p3[0], len(id_p3), id_p3[0:10])
        print(id_np3[0], len(id_np3), id_np3[0:10])
        print('ID_NP3 / NUMBER OF NP3 EVENTS POST SHUFFLE: ', len(id_np3))
    # Aggregate P3 signals together.
    p3 = X[id_p3, :]
    # Aggregate NP3 signals together to a ratio relative to the P3 class events.
    np3 = X[id_np3, :]
    # Aggregate P3 and NP3 events into single data matrix.
    bal_data = np.append(p3, np3, axis=0)
    # Index into P3 / NP3 label locations and append.
    bal_labels = np.append(y[id_p3], y[id_np3])

    if verbose == 1:
        print('P3:NP3 ratio: {0} : {1} / {2} : {3}'.format(1, balance, len(id_p3), len(id_np3)))
        print('Bal Data DIMS: ', bal_data.shape)
        print('Bal Labels DIMS: ', bal_labels.shape)

    return bal_data, bal_labels


def down_S(data, factor, samp_ratekHz, plotter, verbose):
    '''
    Downsamples a signal by a given factor.

    Assumes Trials x Samples.

    Inputs:
        data = eeg input array.
        factor = degree to which data is downsampled by, i.e. factor = 2 would half the signal lenth.
        samp_ratekHz = sample rate of input signal
        plotter = plot (1) for graph of pre and post downsampled signal.
        verbose = 1 : print out info.

    Output:
        Downsampled signal.
        New sample rate after downsampling engaged.

    Example:

    data_A, sampR = pb.down_S(data_A, factor=2, plotter=0, samp_ratekHz=samp_ratekHz, verbose=1)

    '''
    from scipy import signal
    # New Sample Rate.
    neo_samp_ratekHz = samp_ratekHz / factor
    # CRITICAL: Assumes Trials x Samples.
    num_trials = np.shape(data)[0]
    num_samps = np.shape(data)[1]
    print('=====Num Trials: ', num_trials)
    print('=====Num Samps: ', num_samps)
    # Output size of the downsampled array.
    down_factor = np.int(num_samps / factor)
    print('Factor Value: ', down_factor)
    re_X = np.zeros((num_trials, down_factor))
    print('=====re_X DIMS: ', re_X.shape)
    # Utilizing the resample fuction to squash the signal.
    for i in range(num_trials):
        re_X[i, :] = signal.resample(data[i, :], down_factor, t=np.linspace(
            start=0, num=down_factor, stop=num_samps))
    if verbose == 1:
        print('OG Data DIMS: ', data.shape, '|  Resampled Data DIMS: ', re_X.shape)
    # Downsampling plots confirmation.
    if plotter == 1:
        plt.subplot(211)
        plt.title('Pre-Downsampling')
        x_axis = simp_temp_axis(data[0, :], samp_ratekHz)
        plt.plot(x_axis, data[0, :])
        plt.subplot(212)
        plt.title('Post-Downsampling')
        x_axis = simp_temp_axis(re_X[0, :], neo_samp_ratekHz)
        plt.plot(x_axis, re_X[0, :])
        plt.show()
    return re_X, neo_samp_ratekHz


def down_samp(data, factor, samp_ratekHz, plotter, verbose):
    '''
    Downsamples a signal by a given factor.

    Assumes Trials x Samples.

    Inputs:
        data = eeg input array.
        factor = degree to which data is downsampled by, i.e. factor = 2 would half the signal lenth.
        samp_ratekHz = sample rate of input signal
        plotter = plot (1) for graph of pre and post downsampled signal.
        verbose = 1 : print out info.

    Output:
        Downsampled signal.
        New sample rate after downsampling engaged.

    Example:

    pb.down_samp

    '''
    # New Sample Rate.
    neo_samp_ratekHz = samp_ratekHz / factor
    # CRITICAL: Assumes Trials x Samples.
    num_trials = np.shape(data)[0]
    num_samps = np.shape(data)[1]
    # Output size of the downsampled array.
    down_factor = np.int(num_samps / factor)
    re_X = np.zeros((num_trials, down_factor))
    # Utilizing the resample fuction to squash the signal.
    t = np.linspace(start=0, num=down_factor, stop=num_samps, dtype=int)
    t_ind = t - 1
    for i in range(num_trials):
        re_X[i, :] = data[i, t_ind]
    if verbose == 1:
        print('---Downsampling Info')
        print('Num Trials: ', num_trials)
        print('Num Samps: ', num_samps)
        print('Factor Value: ', down_factor)
        print('re_X DIMS: ', re_X.shape)
        print('Input Sampe Rate kHz: ', samp_ratekHz)
        print('New Sample Rate kHz: ', neo_samp_ratekHz)
        print('OG Data DIMS: ', data.shape, '|  Resampled Data DIMS: ', re_X.shape)
    # Downsampling plots confirmation.
    if plotter == 1:
        plt.subplot(211)
        plt.title(
            'Pre-Downsampling | OG Samp Rate: {0}Hz | 1 Sample Per {1}ms'.format(samp_ratekHz * 1000, 1 / samp_ratekHz))
        x_axis = simp_temp_axis(data[0, :], samp_ratekHz)
        plt.plot(x_axis, data[0, :])
        plt.subplot(212)
        plt.title(
            'Post-Downsampling | Neo Samp Rate: {0}Hz | 1 Sample Per {1}ms'.format(down_factor, 1 / neo_samp_ratekHz))
        x_axis = simp_temp_axis(re_X[0, :], neo_samp_ratekHz)
        plt.plot(x_axis, re_X[0, :])
        plt.tight_layout()
        plt.show()
    return re_X, neo_samp_ratekHz


def int_labels(y):
    # Generate and output integer labels.
    y2 = []
    for p in range(len(y)):
        if p == 0:
            y2 = int(y[p])
        else:
            y2 = np.append(y2, int(y[p]))
    return y2


def data_parsing(data, labels, n_splits, train_size, test_size, random_state, multi_chan, verbose):
    '''
    Data splitting for test and train data and labels.

    Can be done for single chan (post channel averaging / or isolated Cz signals),
    or at the multi channel level (however not randomsied).

    Assumes Trials x Samples.

    Inputs:
        data = eeg input array.
        labels = ground truth.
        splits = number of chunks for train / test evaluation, good for checking start vs end of seesion.
        train_per = e.g. 0.85 would give 85% of the data to the train set.
        rand_state = if randmisation engaged (is as default) in sss, then changes random seed.
        multi_chan = 0 : signle chan, 1 : multi-chan splitting.
        verbose = 1 : print out info.

    Output:
        X_train, y_train, X_test, y_test

    Example:

    X_train, y_train, X_test, y_test = pb.data_parsing(data, labels, splits, train_per,
                                                       rand_state, multi_chan)
    '''
    from sklearn.model_selection import StratifiedShuffleSplit

    # Generate split object with sss.
    test_size = 1 - train_size
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=random_state)
    sss.get_n_splits(data, labels)
    print('SSS Split: ', sss)

    data = np.swapaxes(data, 0, 1)
    print('data dims: ', data.shape, 'labels dims: ', labels.shape)

    '----Multi-Chan Train/ Test Parsing------'
    if multi_chan == 'OFF':
        for train_index, test_index in sss.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
    elif multi_chan == 'ON':
        ind = np.int(len(labels)*train_per)
        X_train = data[:, 0:ind]
        X_test = data[:, ind:]
        y_train = labels[0: ind]
        y_test = labels[ind:]
    if verbose == 1:
        print('X train: ', X_train.shape, 'X_test: ', X_test.shape,
              'y train: ', y_train.shape, 'y_test: ', y_test.shape)
    return X_train, y_train, X_test, y_test


def log_reg(data, labels, n_splits, train_size, random_state, multi_chan,
            solver, penalty, max_iter, cross_val_acc, covmat, verbose):
    '''
    Application of Logistic Regression.

    Assumes Trials x Samples.

    Inherently uses data_parsing SSS Split technique / function.

    Inputs:
        data = eeg input array.
        labels = ground truth.
        splits = number of chunks for train / test evaluation, good for checking start vs end of seesion.
        train_size = e.g. 0.85 would give 85% of the data to the train set.
        random_state = if randmisation engaged (is as default) in sss, then changes random seed.
        solver = method of regression.
        multi_chan = 0 : signle chan, 1 : multi-chan splitting.
        penalty = loss function penalty e.g 'l1' or 'l2', only available for some solvers.
        max_iter = number of iterations the log reg takes before converging.
        cross_val_acc = if you want to check performance using cross val (1).
        covmat = compute and print confusion matrix of results, if 1 perform.
        verbose = 1 : print out info.

    Output:
        Performance results.

    Example:
    pb.log_reg(data, labels, n_splits=2, train_size=0.85, random_state=2,
               solver='lbfgs', penalty='l2', cross_val_acc=1, verbose=1)
    '''
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

    # Generate train / test split data.
    test_size = 1 - train_size
    X_train, y_train, X_test, y_test = data_parsing(data, labels, n_splits=n_splits,
                                                    train_size=train_size, test_size=test_size,
                                                    random_state=random_state,
                                                    multi_chan=multi_chan, verbose=1)
    print('X train: ', X_train.shape, 'X_test: ', X_test.shape,
          'y train: ', y_train.shape, 'y_test: ', y_test.shape)

    # Logistic Regression.
    clf = LogisticRegression(random_state=random_state, solver=solver,
                             penalty=penalty, max_iter=1000).fit(X_train, y_train)
    predictions = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    score = clf.score(X_test, y_test)
    if verbose == 1:
        print('Actual Labels: ', y_test)
        print('Predictions:   ', predictions)
        print('Predict Probabilities: \n', pred_proba)
        print('---Standard Testing Score: ', score)
    if covmat == 1:
        from sklearn.metrics import confusion_matrix
        cov_mat = confusion_matrix(y_test, predictions)
        print(cov_mat)
    '---------------------------------------------------------------------------------------'
    cross_val_acc = 1
    if cross_val_acc == 1:
        # Accuracy via cross val.
        cr_data = np.swapaxes(data, 0, 1)
        clf = LogisticRegression(random_state=random_state, solver=solver,
                                 penalty=penalty, max_iter=1000).fit(cr_data, labels)
        cv = StratifiedShuffleSplit(
            n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=random_state)
        accuracy = cross_val_score(clf, cr_data, labels, cv=cv)
        if verbose == 1:
            print('Accuracy', accuracy)
            print('---Cross Val Mean Accuracy: ', np.mean(accuracy) * 100, '%')
            print('Standard Deviation', "%f" % np.std(accuracy))


def lda_(data, labels, split, div, num_comp, meth, covmat, model, plots, import_model, mod_file, verbose):
    '''
    Application of LDA for data discrimation analysis.

    Assumes Trials x Samples.

    Inputs:
        data = data matrix of EEG.
        labels = ground truth labels.
        split = num splits in the Stratified Shuffle Splits for train and test sets.
        div = division in the split between train and test e.g 0.85 == 85% train weighted.
        n_components = dimensions of the embedding space.
        meth = LDA method e.g. 'eigen'
        covmat = compute and print confusion matrix of results, if 1 perform.
        model = return model (if == 1).
        plots = pre and post scale plots toggle (1 == True).
        import_model = defines if a model used in the localization procedue is used for lda (1 == True).
        mod_file = location of lda model.
        verbose = 1 : print out info.

    Output:
        Plots of the TSNE, this analysis is only for visualization.

    Example:

    pb.lda_(data, labels, split, div, num_comp, meth, verbose)

    '''
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedShuffleSplit

    '---Parameters---'
    if split is None:
        split = 2
    if div is None:
        div = 0.85
    if num_comp is None:
        num_comp = 2
    if meth is None:
        meth = 'eigen'

    if import_model == 0:
        test_size = 1 - div
        # Data splitting for test and train.
        sss = StratifiedShuffleSplit(n_splits=split, train_size=div,
                                     test_size=test_size, random_state=2)
        sss.get_n_splits(data, labels)
        for train_index, test_index in sss.split(data, labels):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        clf = LinearDiscriminantAnalysis(solver=meth, shrinkage='auto',
                                         n_components=num_comp).fit(X_train, y_train)
        '---LDA---'
        # Performance.
        if meth == 'eigen':
            print('Explained Covariance Ratio of Components \n: ', clf.explained_variance_ratio_)
        print('Classes: ', clf.classes_)
        predictions = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)
        score = clf.score(X_test, y_test)
        if verbose == 1:
            print('Size of Test Sample: ', len(y_test))
            print('Actual Labels: ', y_test)
            print('Predictions:   ', predictions)
            print('Predict Probabilities: \n', np.round(pred_proba, decimals=2))
            print('Score: ', score)
        if covmat == 1:
            from sklearn.metrics import confusion_matrix
            cov_mat = confusion_matrix(y_test, predictions)
            print(cov_mat)
        if model == 1:
            '---LDA Model across all data---'
            model = LinearDiscriminantAnalysis(solver=meth, shrinkage='auto',
                                               n_components=num_comp).fit(data, labels)
            return model
    elif import_model == 1:
        model = pickle.load(open(mod_file, 'rb'))
        print('Classes: ', model.classes_)
        predictions = model.predict(data)
        pred_proba = model.predict_proba(data)
        score = model.score(data, labels)
        if verbose == 1:
            print('Size of Test Sample: ', len(labels))
            print('Actual Labels: ', labels)
            print('Predictions:   ', predictions)
            print('Predict Probabilities: \n', np.round(pred_proba, decimals=2))
            print('Score: ', score)
        if covmat == 1:
            from sklearn.metrics import confusion_matrix
            cov_mat = confusion_matrix(labels, predictions)
            print(cov_mat)
        return predictions, pred_proba


def tSNE_3D(data, labels, n_components, init, perplexity, learning_rate, multi, verbose):
    '''
    Application of 3D TSNE for visualization of high dimensional data, however setting n_components at 2
    prints a 2D projection on a 3D graphic.

    Reference: https://stackoverflow.com/questions/51386902/t-sne-map-into-2d-or-3d-plot

    Assumes Trials x Samples.

    Inputs:
        data = data matrix of EEG.
        labels = ground truth (NUMERIC).
        n_components = dimensions of the embedding space.
        init = possible options are ‘random’, ‘pca’, and a numpy array of shape (n_samples, n_components)
        perplexity = akin to complexity of the data and following computations.
        learning_rate = degree at which the operation attempts to converge.
        multi = if set at an integer value this is used to control the number of channels used to compute the tSNE.
        verbose = if 1 prints out info on the tSNE applied.

    Output:
        Plots of the TSNE, this analysis is only for visualization.

    Example:

    pb.tSNE_2D(aug_data, labels, n_components=None, perplexities=None, learning_rates=None)

    '''
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    '---Parameters---'
    if n_components is None:
        n_components = 3  # Typically between 5-50.
    if init is None:
        init = 'pca'
    if perplexity is None:
        perplexity = 30  # Typically around 30.
    '---Plot Prep---'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    tsne = TSNE(n_components=n_components, init=init,
                perplexity=perplexity, learning_rate=learning_rate,
                n_iter=10000, n_iter_without_progress=300,
                verbose=verbose).fit_transform(data)

    if multi is None:
        ax.scatter(*zip(*tsne), c=data[:, 0], cmap='RdBu')
    elif multi == 1:
        for i in range(multi):
            ax.scatter(*zip(*tsne), c=data[:, i], cmap='RdBu')
    elif multi == 2:
        '---Label Mapping | RED == P3 | GREEN == NP3---'
        red = labels == 0
        green = labels == 1
        print('tsne red: ', tsne[red, :])
        print('data comp: ', data[:, 0].shape)
        ax.scatter(*zip(*tsne[red, :]))  # c=np.arange(500, 500) # data[0:np.int(len(data)/2), :])
        # , c=np.arange(500, 500) # c=data[np.int(len(data)/2):-1, :])
        ax.scatter(*zip(*tsne[green, :]))
    plt.show()


def tSNE_2D(X, labels, n_components, perplexities, learning_rates):
    '''
    Application of 2D TSNE for visualization of high dimensional data.

    Assumes Trials x Samples.

    Inputs:
        X = data matrix of EEG.
        labels = ground truth labels.
        n_components = dimensions of the embedding space.
        perplexities = akin to complexity of the data and following computations.
        learning_rates = degree at which computations will attempt to converge.

    Output:
        Plots of the TSNE, this analysis is only for visualization.

    Example:

    pb.tSNE_2D(aug_data, labels, n_components=None, perplexities=None, learning_rates=None)

    '''
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    from time import time

    '---Parameters---'
    if n_components is None:
        n_components = 2  # Typically between 5-50.
    if perplexities is None:
        perplexities = [15, 30, 45, 60]  # Typically around 30.
    if learning_rates is None:
        learning_rates = [5, 10, 500, 1000]  # Typically between 10 - 10000
    '---Subplot Prep---'
    (fig, subplots) = plt.subplots(len(learning_rates), len(perplexities) + 1, figsize=(15, 8))

    # Swap Axes.
    X = np.swapaxes(X, 0, 1)
    print('FINAL EEG DIMS: ', X.shape)
    '---Label Mapping | RED == P3 | GREEN == NP3---'
    red = labels == 0
    green = labels == 1
    '---Plotting P3 vs NP3---'
    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    '---TSNE---'
    for j, learning_rate in enumerate(learning_rates):
        for i, perplexity in enumerate(perplexities):
            print('LOC DATA: Perplexity={0} | Learning Rate={1}'.format(
                perplexity, learning_rate))
            ax = subplots[j][i + 1]
            t0 = time()
            tsne = manifold.TSNE(n_components=n_components, init='random',
                                 random_state=0, perplexity=perplexity, learning_rate=learning_rate,
                                 n_iter=10000, n_iter_without_progress=300, verbose=1)
            Y = tsne.fit_transform(X)
            t1 = time()
            print('-------Duration: {0} sec'.format(np.round(t1 - t0), decimals=2))
            ax.set_title('Perplexity={0} | \n Learning Rate={1}'.format(perplexity, learning_rate))
            'Plotting'
            ax.scatter(Y[red, 0], Y[red, 1], c="r")
            ax.scatter(Y[green, 0], Y[green, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
    plt.tight_layout()
    plt.show()


def slice_ext(data_file, data_type, labels_file, markers_file, num_chan, num_emoji, num_seq, out_size, detrend, plotter, verbose):
    '''
    Method for extracting emoji level / event data chunks based on the on-set / offset
    of marker stream pushed label timestamps. This means we extract data only during the
    times at which the stimuli has begun and ended, yielding more rigourous time-corrective
    data values. These emoji-level chunks are interpolated to ensure consistency in
    the temporal separation of data time-points.

    ASSUMES 8 Channels: Fz, Cz, Pz, P4, P3, O1, O2, A2. / [0:7] , important for seq_data parsing.

    # Inputs:

    data_file = the file location containing either the eeg / imp .npz's (Trial-Level).
    data_type = either 'volt' or 'imp' for voltages or impedances data file extraction and slicing.
    labels_file = the file location containing the labels files e.g. 0001_trial_labels.npz (Trial-Level).
    marker_file = the file location containing the marker file (all pushed markers and timestamps) (Session-Level).
    num_chan = number of channels for extraction.
    num_emoji = number of emojis in the stimulus array.
    num_seq = number of sequences in each trial.
    out_size = size of the channel level signal chunk to want returned from the interpolation function.
    detrend = application of polynomial detrending of order 10 (1 == Yes).
    plotter = plot showing the extraction and resampling of one emoji event across all channels using zeroed
              data. The data is not zeroed, only the timestamps, data zeroing is done by prepro function,
              see note below. 1 == plot, 0 == no plot.
    verbose = details of the function operating, 1 == print progress, 0 == do not print.

    # Outputs:

    starts = marker timestamps for all pushed emoji labels occuring at the START of the event (pre-augmenttion).
    ends = marker timestamps for all pushed emoji labels occuring at the END of the event (post-augmentation).
    seq_chk = Extracted data array using marker start and end pushed timestamps, yet to be re-sampled.
    r_data = Aggregate arrays of the extracted and resampled event data, dims = Samples x Channels x Seqs x Trials
    r_times_zer = Aggregate arrays of the ZEROED extracted and resampled event timestamps, dims = Samples x Seqs x Trials
    r_times = 1D array of the extracted non-re-sampled event timestamps for temporal linear session time checking.
    num_trials = number of trials across entire session.


    Example:

    NOTE: the timestamps ARE zeroed, the data is NOT zeroed. The interp function requires the x-axis to be increasing.
    The timestamps from LSL are output in such large system time numbers it cannot reliably detect the increasing,
    or some strange rounding is occuring.

    -Ensure non-zeroed time-stamps are stored, reshaped and plotted to ensure there is cross-session temporal consistency.
    '''

    # Get Trial Data file locations.
    dat_files = pathway_extract(data_file, '.npz', data_type, full_ext=0)
    eeg_files = path_subplant(data_file, np.copy(dat_files), 0)
    if verbose == 1:
        print('EEG Files DIMS: ', np.shape(eeg_files), 'EEG Files: ', eeg_files)
    # Experimental Parameters.
    num_trials = len(eeg_files)
    # Get Labels file locations.
    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)
    # Marker Data.
    markers = np.load(markers_file)
    # Marker Timestamps.
    starts = markers['arr_1']
    ends = markers['arr_3']
    # Marker Labels e.g. '6', or '0', or '1', a string of the actual emoji location augmented..
    mark_start = markers['arr_0']
    mark_end = markers['arr_2']
    # Markers reshape by trials and seqs.
    starts = np.reshape(starts, (num_trials, num_seq, num_emoji))
    ends = np.reshape(ends, (num_trials, num_seq, num_emoji))
    # Aggregate arrays.
    # Samples x Channels x Seq x Trials
    r_data = np.zeros((out_size, num_chan, num_emoji, num_seq, num_trials))
    # Samples x Sequences x Trials : Non-Zeroed.
    r_times = []
    # Aggregate arrays for zeroed timestamps plotting.
    r_times_zer = np.zeros((out_size, num_emoji, num_seq, num_trials))

    for t in range(num_trials):
        # Loading Data.
        data = np.load(eeg_files[t])
        # Loading Labels.
        labels = np.load(lab_files[t])  # .npz containg both labels related files (see below).
        # Matrix containing the order of augmentations for all emoji locations across the trial.
        order = labels['arr_0']
        # Extract Targ Cued for each seqence.
        targs = labels['arr_1']  # List of target cues across the entire trial.
        targ_cue = targs[t]  # List of the nth target cue for the nth trial.
        if verbose == 1:
            print('EEG File_Name', eeg_files[t])
            print('Labels: ', labels)
            print('LABS File_Name', lab_files[t])
            print('Order', order)
            print('Targs', targs)
            print('Targ Cue', targ_cue)
            print('Marker Start Labels', mark_start)
            print('Marker End Labels', mark_end)

        for i in range(num_seq):
            pres_ord = order[i, :]  # List of the nth Sequence's augmentation order from 1 trial.
            # temporal position of target cue augmented during the trial.
            f_index = pres_ord[targ_cue]
            if verbose == 1:
                print('Pres Ord: ', pres_ord)
                print('F Index: ', f_index)
            # EEG Data.
            sequence = 'arr_{0}'.format(i)  # list key for extraction of 1 sequence worth of data.
            # Sequence-Level data parsing only relevent electrodes
            seq_data = data[sequence][:, 0:num_chan]

            '--------DETRENDING--------'
            if detrend == 1:
                chan_ind = ['Fz', 'Cz', 'Pz', 'P4', 'P3', 'O1', 'O2', 'A2']
                seq_data = pol_det(np.copy(seq_data), chan_ind, order=10, plot=False)

            # Sequence-level timestamps from main data array.
            seq_time = data[sequence][:, -1]
            if verbose == 1:
                print('Seq Data DIMS: ', seq_data.shape)
                print('Seq Time DIMS: ', seq_time.shape)
            for j in range(num_emoji):
                # START: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                v_s = starts[t, i, j]
                # Index in timestamp array closest to onset of marker indcating the start of the emoji event.
                str_idx = (np.abs(seq_time - v_s)).argmin()
                # END: Find nearest value of the marker timestamps in the corresponding data timestamp array.
                # Pad to ensure all P3 wave form extracted, taking marker start point and adding 0.5s, indexing to that location in the data array.
                # Just a check to ensure the end marker is not below 0.3s (past peak of the P3 waveform).
                if ends[t, i, j] < starts[t, i, j] + 0.3:
                    v_e = starts[t, i, j] + 0.5
                else:
                    print('Crash Code: End Marker Positioned Before P3 Propogation.')
                    v_e = starts[t, i, j] + 0.5
                # Index in timestamp array closest to onset of marker indcating the end of the emoji event.
                end_idx = (np.abs(seq_time - v_e)).argmin()
                if verbose == 1:
                    print('V_s: ', v_s, 'V_E: ', v_e)
                    print('str_idx : ', str_idx, 'end_idx: ', end_idx)
                # Indexing into data array to extract currect P300 chunk.
                seq_chk = seq_data[str_idx: end_idx, :]
                # Indexing into timestamp array to extract currect P300 chunk timestamps.
                if verbose == 1:
                    print('Str Idx: ', str_idx, 'End Idx: ', end_idx)
                seq_temp = seq_time[str_idx: end_idx]  # Non-Zeroed Timestamps @ Sequence Level.
                r_times = np.append(r_times, seq_temp)  # Non-Zeroed Timestamps @ Trial Level.
                # Zeroed Timestamps @ Sequence Level.
                seq_temp_zer = seq_temp - seq_temp[0]
                # Resampling Interpolation Method @ Channel Level, using zeroed timestamp values.
                r_data[:, :, j, i, t], r_times_zer[:, j, i, t] = interp2D(
                    seq_chk, seq_temp_zer, output_size=out_size, plotter=0, verbose=0)
                'Verbose Details of operation.'
                if verbose == 1:
                    print('V_s: ', v_s, 'Start IDX: ', str_idx, 'V_e: ', v_e, 'End IDX: ', end_idx)
                    print('Diff in time between Start and End: ', v_e - v_s)
                    print('Emoji: {0} | Seq: {1}'.format(j + 1, i + 1),
                          'Seq_Chk Dims: ', seq_chk.shape)
                    print('r_data DIMS: ', r_data.shape, 'r_times DIMS: ', r_times.shape)
                    'Zeroed Data Section for Plotting.'
    if plotter == 1:
        plt.plot(r_times_zer[:, 0, 0, 0], r_data[:, 0, 0, 0, 0])
        plt.title(
            'Resampled Timestamps (X Axis) and Data (Y Axis) for 1st Channel in 1st Sequence in 1st Trial')
        plt.show()
        plt.plot(r_times)
        plt.title(
            'Non-Resampled Timestamps to check ascending and consistent session progression in temporal terms.')
        plt.show()
    return starts, ends, seq_chk, r_data, r_times_zer, r_times, num_trials


def time_check(data_file, markers_file):
    '''
    Compares timestapms collected by the eeg / imp and marker streams to ensure maxinmal alignment.
    Plots the onset and offset of pushed marker stream samples against the timestamp eeg / imp stream values.
    Also, plots data using data stream timstamp axis vs pre-gen perfect axis to illustrate temporal acquisition inconsistency.

    # Inputs;

    data_file = specific data .npz file for a single trial, assumes 14 channels, 7 actual electrodes.

    makers_file = marker_data.npz containing the pushed marker labels for the entire session period.

    # NO OUTPUTS.

    # Example:

    data_file = '..//Data_Aquisition/Data/voltages_t0001_s1_190917_102243806411.npz'
    markers_file = '..//Data_Aquisition/Data/marker_data.npz'

    time_check(data_file, markers_file)

    '''

    data = np.load(data_file)
    markers = np.load(markers_file)
    # EEG Data.
    sequence = 'arr_0'
    seq1_data = data[sequence][:, 0:7]
    print('Seq Data DIMS: ', seq1_data.shape)
    # EEG Timestamps.
    seq1_time = data[sequence][:, -1]
    print('Seq Time DIMS: ', seq1_time.shape)
    print('First Data Time Stamp: ', seq1_time[0], ': ', 0)
    print('Last Data Time Stamp: ', seq1_time[-1], ': ', seq1_time[-1] - seq1_time[0])
    # Marker Data.
    seq_mark_end = markers['arr_3']
    seq_mark_str = markers['arr_1']
    print('Seq1 Mark DIMS: ', seq_mark_str.shape)
    print('1st Mark Stamp: ', seq_mark_str[0])
    # Diff between 1st EEG Timestamp and 1st Marker Timestamp.
    print('Data Marker Offset: ', seq_mark_str[0] - seq1_time[0])

    for i in range(len(seq_mark_str)):
        print('Length Mark Collection Emoji {0}: '.format(
            i + 1), seq_mark_end[i] - seq_mark_str[i], 'Start: ', seq_mark_str[i], 'End: ', seq_mark_end[i])

    'Plots'
    # Plot EEG Data Timestamps.
    plt.plot(seq1_time)
    num_emojis = 7
    print('1st Sequence Start Times: ', seq_mark_str[0:6])
    mark1 = np.zeros(len(seq1_time))
    mark2 = np.zeros(len(seq1_time))

    for i in range(num_emojis):
        # Plot Marker Start Times.
        mark1[:] = seq_mark_str[i]
        print('Start Time: ', seq_mark_str[i])
        plt.plot(mark1)
        # Plot Marker End Times.
        mark2[:] = seq_mark_end[i]
        print('End Time: ', seq_mark_end[i])
        plt.plot(mark2)
    plt.title('Marker Start and End Points Overlaid on EEG OR IMP Data Timestamps.')
    plt.show()

    # Data with Data Timestamp Axis.
    plt.plot(seq1_time, seq1_data[:, 0])
    plt.title('Data with Data Timestamp Axis')
    plt.show()
    # Data with Pre-Gen Timestamp Axis.
    gen_time = np.arange(len(seq1_data[:, 0]))
    plt.plot(gen_time, seq1_data[:, 0])
    plt.title('Data with Pre-Gen Timestamp Axis')
    plt.show()

    # Find nearest value of the marker start timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_str[0]
    idx = (np.abs(arr - v)).argmin()
    print('Start Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])
    # Find nearest value of the marker end timestamps in the corresponding data timestamp array.
    arr = seq1_time
    v = seq_mark_end[6]
    idx = (np.abs(arr - v)).argmin()
    print('End Idx: ', idx, 'Idx of Seq Time: ',
          seq1_time[idx], 'Idx of Seq Data: ', seq1_data[idx, 0])


def binary_labeller(labels, verbose):
    '''
    Option for binary labelling of te data as either containing P3 ('1') or containing NP3 ('0').

    Assumes 1D array of integers computed via the spatial labeller output as specified in the script.

    Verbose: if 1 prints some examples from the ouput, if 0, no printing.

    '''
    y = labels
    for i in range(len(y)):
        if int(y[i]) != 0:
            y[i] = '0'
        elif int(y[i]) == 0:
            y[i] = '1'
    if verbose == 1:
        print('Base Normalized Y Labels: ', y[0:10])
    return y


def spatial_labeller(labels_file, num_emoji, num_seq, verbose):
    '''

    Method of extracting spatial labels from the flash order of emoji augmentations.
    e.g. Sequence = [3, 5, 1, 6, 0, 2, 4] | Target Cue = 3, meaning that the 4th emoji
    underwent an augmentation, as from this sequence above you can see that the 4th
    emoji was augmented 2nd (1).

    Labelling the augmentation events spatially involves describing each emoji in terms
    of distance from the target emoji which is being attended/ fixated.

    Sequence = [3, 5, 1, 6, 0, 2, 4] , would become [2, 1, 0, 1, 2, 3, 4], with zero indicating
    the emoji location cued and value labels emanating from this location increasing
    as a function of distance from this spatial location.

    # Inputs:

    labels_file = Simply specify the file location of the trial labels.
                  e.g. '..//Data_Aquisition/Data/Labels/'
    num_emoji =  number of emoji in the stimulus array.

    num_seq = number of sequences in each trial.

    verbose = specify if you want to print details of labelling (1 == Yes, 0 == No).

    # Outputs:

    sp_labels = a 1D label array for all event chunks segmented by SegBoy (around 500ms each).

    # Example:

    sp_labels = pb.spatial_labeller(labels_file, num_emoji, num_seq, verbose=0)

    OR

    sp_labels = spatial_labeller('..//Data_Aquisition/Data/Labels/', 7, 5, 0)

    '''

    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)

    # Experimental Parameters.
    num_trials = len(lab_files)
    if verbose == 1:
        print('Num Trials: ', num_trials)

    # Aggregate PreFrom Array @ Trial Level.
    sp_labels = []

    for t in range(num_trials):
        # Loading Labels.
        labels = np.load(lab_files[t])
        order = labels['arr_0']
        # Extract Targ Cued for each seqence.
        targs = labels['arr_1']
        targ_cue = targs[t]
        if verbose == 1:
            print('Labels: ', labels)
            print('LABS File_Name', lab_files[t])
            print('Order', order)
            print('Targs', targs)
            print('Targ Cue', targ_cue)
            # Spatial Labelling.
            print('------Spatial Labeller')
        # Aggregate Preform Array.
        fin_sp = []
        for j in range(num_seq):
            sp_pres = []
            targ_cue = targs[j]
            if verbose == 1:
                print('Targ Cue: ', targ_cue)
            for i in range(num_emoji):
                pin = np.array2string(np.abs(i - targ_cue))
                if i == 0:
                    sp_pres = pin
                else:
                    sp_pres = np.append(sp_pres, pin)
                if i and j and t == [0, 0, 0]:
                    sp_labels = pin
                else:
                    sp_labels = np.append(sp_labels, pin)
            sp_pres = np.expand_dims(sp_pres, axis=1)
            if j == 0:
                fin_sp = sp_pres
            else:
                fin_sp = np.append(fin_sp, sp_pres, axis=1)
            if verbose == 1:
                print('Sequence {}: '.format(j + 1), sp_pres.shape, ' \n', sp_pres)
        if verbose == 1:
            print('Trial {}: '.format(t + 1), fin_sp.shape, ' \n', fin_sp)
            # Aggreaate into 1D tp_labels array across the Seesion.
            print('Spatial Labels 1D Array: ', sp_labels.shape, ' \n', sp_labels)
    return sp_labels


def temporal_labeller(labels_file, num_emoji, num_seq, verbose):
    '''
    Method of extracting temporal labels from the flash order of emoji augmentations.
    e.g. Sequence = [3, 5, 1, 6, 0, 2, 4] | Target Cue = 3, meaning that the 4th emoji
    underwent an augmentation, as from this sequence above you can see that the 4th
    emoji was augmented 2nd (1).

    Labelling the augmentation events temporal involves describing each emoji in terms
    of distance in TIME from the target emoji which is being attended/ fixated.

    Sequence = [3, 5, 1, 6, 0, 2, 4] , would become [+2, +4, 0, +5, -1, +1, +3], with zero
    indicating the emoji location cued and value labels assigned to other locations
    differing as a function of temporal distance from this timed event.

    # Inputs:

    labels_file = Simply specify the file location of the trial labels.
                  e.g. '..//Data_Aquisition/Data/Labels/'

    num_emoji =  number of emoji in the stimulus array.

    num_seq = number of sequences in each trial.

    verbose = specify if you want to print details of labelling (1 == Yes, 0 == No).

    # Outputs:

    tp_labels = a 1D label array for all event chunks segmented by SegBoy (around 500ms each).

    # Example:

    tp_labels = pb.temporal_labeller(labels_file, num_emoji, num_seq, verbose=0)

    OR

    tp_labels = temporal_labeller('..//Data_Aquisition/Data/Labels/', 7, 5, verbose=0)

    '''
    # Get Labels file locations.
    labels_file = labels_file
    grn_files = pathway_extract(labels_file, '.npz', 'trial', full_ext=0)
    lab_files = path_subplant(labels_file, np.copy(grn_files), 0)
    if verbose == 1:
        print('Lab Files DIMS: ', np.shape(lab_files), 'Lab Files: ', lab_files)

    # Experimental Parameters.
    num_trials = len(lab_files)
    if verbose == 1:
        print('Num Trials: ', num_trials)

    # Aggregate PreFrom Array @ Trial Level.
    tp_labels = []
    for t in range(num_trials):
        if verbose == 1:
            print('----Trial: ', t + 1)
        # Loading Labels.
        labels = np.load(lab_files[t])
        order = labels['arr_0']
        # Extract Targ Cue for each seqence.
        targs = labels['arr_1']
        targ_cue = targs[t]
        for j in range(num_seq):
            tp_pres = []
            pres_ord = order[j, :]
            f_index = pres_ord[targ_cue]
            print('F INDEX: ', f_index)
            if verbose == 1:
                print('Pres Order: ', pres_ord)
                print('Targ Cue: ', targ_cue)
            for i in range(num_emoji):
                pin = np.array2string(f_index - pres_ord[i])
                # Sequence Level Aggregation.
                if i == 0:
                    tp_pres = pin
                else:
                    tp_pres = np.append(tp_pres, pin)
                # Cross Trial Aggregation.
                if i and j and t == [0, 0, 0]:
                    tp_labels = pin
                else:
                    tp_labels = np.append(tp_labels, pin)
            tp_pres = np.expand_dims(tp_pres, axis=1)
            if j == 0:
                # Aggregate Sequence Labels to Trial Labels.
                fin_tp = tp_pres
            else:
                fin_tp = np.append(fin_tp, tp_pres, axis=1)
            if verbose == 1:
                print('Sequence {}: '.format(j + 1), tp_pres.shape, ' \n', tp_pres)
    if verbose == 1:
        print('Trial {}: '.format(t + 1), fin_tp.shape, ' \n', fin_tp)
        # Aggreaate into 1D tp_labels array across the Seesion.
        print('Temporal `Labels 1D Array: ', tp_labels.shape, ' \n', tp_labels)
    return tp_labels


def interp2D(data, timestamps, output_size, plotter, verbose):
    # Resamples 2D data matrices of Samples x Channels via interpolation to produce uniform output matrices of output size x channels.

    # Calcualte number of chans.
    a, b = np.shape(data)
    num_chans = np.minimum(a, b)
    # Gen place-holder for resampled data.
    r_data = np.zeros((output_size, num_chans))
    r_time = np.linspace(0, output_size * 0.002, output_size)

    for k in range(num_chans):
        # Interpolate Data and Sub-Plot.
        yinterp = np.interp(r_time, timestamps, data[:, k])
        # Aggregate Resampled Channel Data and Timestamps.
        r_data[:, k] = yinterp

        # Plots
        if plotter == 1:
            # Sub-Plot Non-Resampled Channel Chk
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, data[:, k])
            plt.title('Orignal Signal With Inconsistent Timestamps.')
            # Sub-Plot Resampled Channel Chk
            plt.subplot(2, 1, 2)
            plt.plot(r_time, yinterp)
            plt.title('Signal Post-Interpolation Method Resampling.')
            plt.show()
        if verbose == 1:
            print('OG DIMS: ', data[:, k].shape,
                  'Resampled Dims: ', yinterp.shape)
            og_max = np.amax(np.diff(timestamps))
            og_min = np.amin(np.diff(timestamps))
            neo_max = np.amax(np.diff(r_time))
            neo_min = np.amin(np.diff(r_time))
            print('OG Min: ',  og_min,
                  'Resampled Min: ', neo_min)
            print('OG Max: ', og_max,
                  'Resampled Max: ', neo_max)
            print('OG  Range: ', (og_max - og_min),
                  'Resampled Range: ', (neo_max - neo_min))

    return r_data, r_time


def is_odd(num):
    return num % 2


def ranger(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def sess_plot(data, label, ses_tag, num_trl_per_sess):
    # Plot all P3s and NP3s per session.
    # Assumes Sampes, Channels, Trials.
    time = np.arange(0, 500, 2)
    u, p3_ind = np.unique(ses_tag, return_index=True)
    num_trials = data.shape[0]
    p3_ind = np.append(p3_ind, [p3_ind[-1] + num_trl_per_sess])
    np3_ind = p3_ind + np.int(num_trials / 2)
    print('UNQ: ', u.shape, u)
    print('P3 Index: ', p3_ind.shape, p3_ind)
    print('NP3 Index: ', np3_ind.shape, np3_ind)
    for i in range(len(u)):
        plt_p3 = np.average(np.squeeze(data[p3_ind[i]:p3_ind[i + 1], :, :]), axis=0)
        plt_np3 = np.average(np.squeeze(data[np3_ind[i]:np3_ind[i + 1], :, :]), axis=0)
        print('Session {0} | P3 mV Range: {1} / NP3 mV Range: {2}'.format(i +
                                                                          1, ranger(plt_p3), ranger(plt_np3)))
        # Plot Legends.
        p3_p, = plt.plot(time, plt_p3, label='P3')
        np3_p, = plt.plot(time, plt_np3, label='NP3')
        plt.title('Session: {} Signal Averages '.format(i + 1))
        plt.legend([p3_p, np3_p], ['P3', 'NP3'])
        plt.show()
    return u, p3_ind, np3_ind


def rand_data(data, num_trls, num_samps, num_chans):
    # Generate purely randomised data.
    data_ = np.random.rand(num_trls, num_samps, num_chans)
    return data_


def uni_100(data, label, num_trls, num_samps, num_chans, n_con, zer_m, plotter):
    data_ = np.zeros((num_trls, num_samps, num_chans))
    print('Num Samps: ', num_samps, 'Noise: ', n_con, 'Data DIMS: ', data.shape)
    # Generate purely uniform data.
    for i in range(num_trls):
        if label[i] == 0:
            # Create sine wave + add noise.
            window = signal.cosine(num_samps)
            noise = np.random.uniform(0, n_con, num_samps)
            if n_con > 0:
                waveform = window+noise
                waveform = np.expand_dims(waveform, axis=-1)
            elif n_con == 0:
                waveform = window
                waveform = np.expand_dims(waveform, axis=-1)
            data_[i, :, :] = waveform
            if plotter == 1:
                # Plot differences between original signal ('window'), generated noise and the combined waveform.
                win_p, = plt.plot(window, label='Window')
                noise_p, = plt.plot(noise, label='Noise')
                wave_p, = plt.plot(waveform, label='Waveform')
                plt.legend([win_p, noise_p, wave_p], ['Window', 'Noise', 'Waveform'])
                plt.title('Comparing Raw Signal, Noise and Waveform - A Curve')
                plt.show()
        elif label[i] == 1:
            # Create flat signal at zero + add noise.
            window = np.ones(num_samps)
            noise = np.random.uniform(0, n_con, num_samps)
            if n_con > 0:
                waveform = window+noise
                waveform = np.expand_dims(waveform, axis=-1)
            elif n_con == 0:
                waveform = window
                waveform = np.expand_dims(waveform, axis=-1)
            data_[i, :, :] = waveform
            if plotter == 2:
                # Plot differences between original signal ('window'), generated noise and the combined waveform.
                win_p, = plt.plot(window, label='Window')
                noise_p, = plt.plot(noise, label='Noise')
                wave_p, = plt.plot(waveform, label='Waveform')
                plt.legend([win_p, noise_p, wave_p], ['Window', 'Noise', 'Waveform'])
                plt.title('Comparing Raw Signal, Noise and Waveform - B Flat')
                plt.show()
    if zer_m == 1:
        data_ = zero_mean(np.squeeze(data_))
        data_ = np.expand_dims(data_, axis=2)
    if plotter == 1:
        # eeg_series = temp_axis(data, 500)
        raw, = plt.plot(data[400, :, :], label='Raw')
        noised, = plt.plot(data_[400, :, :], label='Noised')
        plt.legend([raw, noised], ['Raw', 'Noised'])
        plt.title('Comparing Raw Signal and Noised Waveform')
        plt.show()
    return data_


def net_sets_parser(data, label, train_per, val_per, test_per):

    # Add new singleton dimesion.
    # Input Dims: Trials, Samples, Channels.
    # Expects: Trials, Singleton, Channels, Samples.
    data = np.swapaxes(np.copy(data), 1, 2)
    data = np.expand_dims(np.copy(data), axis=1)
    print('HERE DATA DIMS: ', data.shape)

    total = np.shape(data)[0]
    tr_dv = np.int(total*train_per)
    vl_dv = np.int(tr_dv + (total*val_per))
    te_dv = np.int(vl_dv + (total*test_per))

    'Train'
    X_train = data[0:tr_dv, :, :, :]
    X_train = X_train.astype('float32')
    print('X_train Dims: ', np.shape(X_train))
    y_train = label[0:tr_dv]
    y_train = y_train.astype('float32')
    print('y_train Dims: ', np.shape(y_train))

    'Val'
    X_val = data[tr_dv:vl_dv, :, :, :]
    X_val = X_val.astype('float32')
    print('X_val Dims: ', np.shape(X_val))
    y_val = label[tr_dv:vl_dv]
    y_val = y_val.astype('float32')
    print('y_val Dims: ', np.shape(y_val))

    'Test'
    X_test = data[vl_dv:te_dv, :, :, :]
    X_test = X_test.astype('float32')
    print('X_test Dims: ', np.shape(X_test))
    y_test = label[vl_dv:te_dv]
    y_test = y_test.astype('float32')
    print('y_test Dims: ', np.shape(y_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepro(eeg, samp_ratekHz, zero, ext, elec, filtH, hlevel, filtL, llevel, notc, notfq, ref_ind, ref, avg):
    # Assumes Samples x Trials.
    'ZERO: mornalize data with zero meaning.'
    if zero == 'ON':
        eeg = zero_mean(np.copy(eeg))
    '---------------------------------------------------'
    'EXTRACT: relevant electrodes:  0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Reference: Guger (2012): Dry vs Wet Electrodes | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3345570/'
    # Ensure reference electrode A2 is extracted by calculating the position in the array.
    # A2 ref_ind always -7, as there are 6 variables at end: ACC8, ACC9, ACC10, Packet, Trigger & Time-Stamps.
    if ext == 'INC-Ref':
        all_eeg = eeg[:, [0, 1, 2, 3, 4, 5, 6]]
        grab = np.append(elec, ref_ind)
        eeg = np.squeeze(eeg[:, [grab]])
    if ext == 'NO-Ref':
        all_eeg = eeg[:, [0, 1, 2, 3, 4, 5, 6]]
        eeg = np.squeeze(eeg[:, [elec]])
    '------------------------------------------------------'
    'REFERENCING: using REF electrode.'
    if ref == 'A2':
        eeg = referencer(np.copy(eeg), -1)
    elif ref == 'AVG':
        eeg = avg_referencer(np.copy(eeg), all_eeg)
    '---------------------------------------------------'
    'FILTERING: highpass filter.'
    if filtH == 'ON':
        eeg = butter_highpass_filter(np.copy(eeg), hlevel, 500, order=5)
    '---------------------------------------------------'
    'FILTERING: lowpass filter.'
    if filtL == 'ON':
        # Fix by adding singleton dimension.
        if len(np.shape(eeg)) == 1:
            eeg = np.expand_dims(np.copy(eeg), axis=1)
        eeg = butter_lowpass_filter(np.copy(eeg), llevel, 500, order=5)
    '----------------------------------------------------'
    'FILTERING: 50Hz notch filter.'
    if notc == 'NIK':
        samp_rateHz = samp_ratekHz * 1000
        eeg = notchyNik(np.copy(eeg), Fs=samp_rateHz, freq=notfq)
    elif notc == 'NOC':
        eeg = notchy(eeg, 500, notfq)
    elif notc == 'LOW':
        eeg = butter_lowpass_filter(np.copy(eeg), notfq, 500, order=5)
    '---------------------------------------------------'
    'AVERAGING: cross-channels.'
    if avg is True:
        eeg = np.average(np.copy(eeg), axis=1)
        eeg = np.expand_dims(eeg, axis=1)
    '---------------------------------------------------'
    return eeg


def notchy(data, Fs, freq):
    'Notch Filter at 50Hz using the IIR: forward-backward filtering (via filtfilt)'
    'Requies Channel x Samples orientation.'

    'Example:'
    # grnd_data = notchyNik(a2_data, Fs=250)

    import mne
    data = np.swapaxes(data, 0, 1)
    filt_data = mne.filter.notch_filter(
        data, Fs=Fs, freqs=freq, method='fir', verbose=False, picks=None)
    # print('1st Data Value: ', data[0, 0], '1st Grounded Value: ', filt_data[0, 0])
    filt_data = np.swapaxes(filt_data, 0, 1)
    return filt_data


def notchyNik(data, Fs, freq):
    from scipy import signal
    fs = Fs
    Q = 30.0  # Quality factor
    w0 = freq/(fs/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)

    chans = np.amin(np.shape(data))
    for i in range(chans):
        input_data = signal.filtfilt(b, a, np.copy(data[:, i]))
        data[:, i] = input_data
    return data


def freqy(data, fs):
    'FFT of 1st channel in eeg_data.'
    from scipy import fftpack
    x = data[:, 0]
    fft_data = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * fs
    return fft_data, freqs


def sess_tagger(sub_path, i):
    sess_ = sub_path[i]
    sess_ = np.copy(sess_[-3])
    sess_tag = sess_.astype(int)
    return sess_tag


def sub_tagger(sub_path, i):
    sub_ = sub_path[i]
    sub_ = np.copy(sub_[25:30])
    # sub_tag = sub_.astype(np.int)
    sub_tag = sub_
    return sub_tag


def sub_tag_2(sub_path, i):
    # Provides subject tagging.
    sub_tag = sub_tagger(sub_path, i)
    sub = np.array2string(sub_tag)
    # Isolate numerical elements.
    sub = sub[1:6]
    # Preform of subject indicator result.
    res = []
    for i in range(len(sub)):
        # If the LEADING element is '0' we want to skip that.
        if sub[i] == '0':
            # If result is empty, keep it empty.
            if res == []:
                res = []
        else:
            # Values < 10.
            # Once element changes from '0' we grab that as the sub number.
            if i == len(sub)-1:
                if res == []:
                    # If the leading values is NOT '0' AND our res preform has NOT been filled, then append.
                    res = sub[i]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 10.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-2:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 100.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-3:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
            # Vales => 1000.
            # Once element changes from '0' we grab that and the rest of the values.
            if i == len(sub)-4:
                if res == []:
                    res = sub[i:]
                    res = np.copy(res)
                    print(res, type(res))
    return res


def expSubSessParser(data, label, all, exp, sub, seord, num_trls, exp_q, sub_q, seord_q):

    data_ = []
    label_ = []

    if all == 1:
        exp_q = np.unique(exp)
        sub_q = np.unique(sub)
        seord_q = np.unique(seord)
        data_ = data
        label_ = label
        exp_ = exp
        seord_ = seord
        sub_ = sub
    elif all == 0:
        for p in range(num_trls):
            if exp[p] in exp_q and seord[p] in seord_q and sub[p] in sub_q:
                if p == 0:
                    data_ = data[p, :, :]
                    data_ = np.expand_dims(data_, axis=0)
                    label_ = label[p]
                    exp_ = exp[p]
                    seord_ = seord[p]
                    sub_ = sub[p]
                elif p != 0:
                    if data_ == []:
                        data_ = data[p, :, :]
                        data_ = np.expand_dims(data_, axis=0)
                        label_ = label[p]
                        exp_ = exp[p]
                        seord_ = seord[p]
                        sub_ = sub[p]
                    else:
                        data_ = np.append(data_, np.expand_dims(data[p, :, :], axis=0), axis=0)
                        label_ = np.append(label_, label[p])
                        exp_ = np.append(exp_, exp[p])
                        seord_ = np.append(seord_, seord[p])
                        sub_ = np.append(sub_, sub[p])
    return data_, label_, exp_, seord_, sub_


def basics(time, signal):
    maxi = np.max(signal)
    mini = np.min(signal)
    return[maxi, mini]


def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def passer(signal, low_cut, high_cut, fs):
    'Finally, what is the best filter for P300 detection? Laurent Bougrain, Carolina Saavedra, Radu Ranta'
    'https://hal.inria.fr/hal-00756669/document'

    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    # band_data = np.squeeze(passer(x, grnd_data, 1, 40))
    num_samps, num_chans = np.shape(signal)
    band_data = np.zeros((num_samps, num_chans))

    for i in range(num_chans):
        band_data[:, i] = butter_bandpass_filter(signal[:, i], low_cut, high_cut, fs, order=5)
    return band_data


def butter_lowpass(cutoff, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    'Example: y = butter_lowpass_filter(data, cutoff, fs, order)'
    'Band-pass filter assumes sharper transitions from the pass band to the stop band as the order of the filter increases'
    from scipy.signal import lfilter

    # print('Iterator: ', (np.amin(np.shape(data))))

    for i in range(np.amin(np.shape(data))):
        low_data = data[:, i]
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, low_data)
        data[:, i] = y
    return data


def butter_highpass(cutoff, fs, order=5):
    from scipy.signal import butter
    'Band-pass filter assumes sharper transitions from the pass band to the stop band as the order of the filter increases.'
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import lfilter

    # print('Iterator: ', (np.amin(np.shape(data))))

    if data.ndim == 1:
        b, a = butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        data = y
    elif data.ndim == 2:
        for i in range(np.amin(np.shape(data))):
            # print('---------------Iterator: ', (np.amin(np.shape(data))))
            # print('---------------Data DIMS: ', np.shape(data))
            high_data = data[:, i]
            b, a = butter_highpass(cutoff, fs, order=order)
            y = lfilter(b, a, high_data)
            data[:, i] = y
    return data


def low_pass_grnd(data, Fs):
    from scipy.signal import freqs

    'Example: '
    # grnd_data = low_pass_grnd(a2_data, 250)

    # Filter requirements.
    order = 8
    fs = Fs      # sample rate, Hz
    cutoff = 50  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqs(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Generate time-stamp values.
    T = 5.0        # seconds
    n = int(T * fs)  # total number of samples
    t = np.linspace(0, T, n, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)
    print('1st Data Value: ', data[0, 0], '1st Grounded Value: ', y[0, 0])
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    # plt.show()
    return y


def referencer(data, grndIdx):
    'Substraction referencing method, commonly use A2 for P300 pre-processing.'
    'Data = Raw EEG time-series.'
    'grndIdx = Channel Index in data tensor of ground electrode.'
    'Requires a 2D tensor for use.'
    'Assumes data in Samples x Channels orientation.'

    'Example: '
    samp, chan = np.shape(data)
    # Isolate Ground Channel
    grnd = data[:, grndIdx]
    # Remove Ground Channel from main data.
    data = np.delete(data, grndIdx, axis=1)
    # Pre-assign Ground Data Array.
    grnd_data = np.zeros((samp, chan - 1))
    for i in range(chan - 1):
        grnd_data[:, i] = data[:, i] - grnd
    return grnd_data


def avg_referencer(eeg, all_eeg):
    'Average referencing across all electrodes.'
    'ext = Relevant EEG electrodes.'
    'all_eeg = All electrodes in relevant montage sampled from: 0) Fz, 1) Cz, 2) Pz, 3) P4, 4) P3, 5) O1, 6) O2, 7) A2.'
    'Requires a 2D tensor for use.'
    'Assumes data in Samples x Channels orientation.'

    'Example:'
    #  eeg = avg_referencer(np.copy(eeg), all_eeg)

    print('EEG Pre AVG REF DIMS: ', eeg.shape)

    # 1st Remove the A2 electrode by deleting the final column of data.
    eeg = np.delete(eeg, (-1), axis=1)
    dims = eeg.shape[1]

    # Generate average reference from eeg montage.
    avg_ref = np.average(all_eeg, axis=1)
    print('AVG REF DIMS: ', avg_ref.shape)

    # Subtract average reference from each eeg channel.
    if dims >= 1:
        for i in range(dims):
            chan_data = eeg[:, i]
            chan_data = chan_data - avg_ref
            # chan_data = np.expand_dims(chan_data, axis=1)
            eeg[:, i] = chan_data
    else:
        eeg = eeg - avg_ref

    # Data Dimesions Check:
    print('Post AVG REF DIMS: ', eeg.shape)
    return eeg


def subplot_dual(dataA, dataB, type, title):
    'Subplotting of 2 data arrays.'
    'Both must use same orientation.'
    'Type: Referenced data plot = 0, 7 Chan Plot plot = 1, 2 Chan Occ Plot = 2.'
    'Title: Name OG Data vs Transformed data.'

    'Example:'
    # subplot_dual(data, ref_data, 0, 'Raw vs Referenced Data')

    f, axarr = plt.subplots(2, 1)
    if type == 0:
        labelsA = ('Fp1', 'F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    elif type == 1:
        labelsA = ('F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    linesA = axarr[0].plot(dataA)
    axarr[0].legend(linesA, labelsA)
    labelsB = ('F3', 'Fz', 'C3', 'Cz', 'Pz', 'P4', 'P3', 'C4', 'A2')
    linesB = axarr[1].plot(dataB)
    axarr[1].legend(linesB, labelsB)
    f.suptitle(title, fontsize=16)


def subplotter(dataA, dataB, time_axis, title):
    'Subplotting of 2 data arrays.'
    'Both must use same orientation.'
    'Type: Referenced data plot = 0, 7 Chan Plot plot = 1, 2 Chan Occ Plot = 2.'
    'Title: Name OG Data vs Transformed data.'

    'Example:'
    # subplotterl(data, ref_data, 'Raw vs Referenced Data')

    f, axarr = plt.subplots(2, 1)
    axarr[0].plot(time_axis, dataA)
    axarr[1].plot(time_axis, dataB)
    f.suptitle(title, fontsize=16)
    plt.show()


def zero_mean(data):
    'Zeros Data, accepts orientation Samples x Channels.'
    a, b = np.shape(data)
    # Preform zero array.
    zero_data = np.zeros((a, b))
    for i in range(b):
        zero_data[:, i] = data[:, i] - np.mean(data[:, i])
    return zero_data


def zerodat(data):
    'Zeros Data, accepts orientation Samples x Channels.'
    'Zero data using 1st value channel sample subtraction'
    'Therefore, signal begins at zero baseline.'

    'Exmaple: '
    # zero_data = zerodat(band_data)

    a, b = np.shape(data)
    # Preform zero array.
    zero_data = np.zeros((a, b))
    for i in range(b):
        zero_data[:, i] = data[:, i] - data[0, i]
    return zero_data


def zero_std(data):
    'Divide by Standard Deviation from each channel.'

    a, b = np.shape(data)
    # Preform zero array.
    zero_std = np.zeros((a, b))
    for i in range(b):
        # Get Channel std.
        std_x = data[:, i].std()
        sub_std = np.zeros(a) + std_x
        zero_std[:, i] = data[:, i] / sub_std

    return zero_std


def scale(x, out_range=(0, 30)):
    # Conversion to mV range.
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def scaler2D(data):
    from sklearn.preprocessing import MinMaxScaler
    # Grab data dims: Trials x Samples x Channels
    tr, sm, ch = np.shape(data)

    for j in range(tr):
        for i in range(ch):
            # Extract channel Data.
            chan_data = data[j, :, i]
            # Format with additional singleton dimension.
            chan_data = np.expand_dims(chan_data, axis=1)
            # Initialize the scaler function.
            scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
            # Fit the data.
            scaler.fit(chan_data)
            # Transform the data.
            clean_data = scaler.transform(chan_data)
            # Re-format for appending.
            clean_data = np.squeeze(clean_data)
            data[j, :, i] = clean_data
    return data


def scaler1D(data, low, high):
    from sklearn.preprocessing import MinMaxScaler
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(copy=True, feature_range=(low, high))
    scaler.fit(data)
    data = scaler.transform(data)
    return data


def power(stop_sig, non_stop_sig, plot, title):
    'Example: '
    # freqs, Pxx_spec = power(avg_data, 0, 'Sing Trial')
    # print(freqs[17:24])  # For relevant frequenies extraction.
    # print(Pxx_spec[17:24])  # For relevant psd value extraction.

    from scipy import signal
    # import matplotlib.mlab as mlab
    'Stop Sig'
    freqs, psd = signal.welch(stop_sig)
    f, Pxx_spec = signal.periodogram(stop_sig, 250, 'flattop', scaling='spectrum')
    dt = 0.004  # Because 1 / 0.004 = 250
    Pxx, freqs = plt.psd(stop_sig, 256, 1 / dt, label='Stop Signal PSD')
    'Non-Stop Sig'
    freqs, psd = signal.welch(non_stop_sig)
    f, Pxx_spec = signal.periodogram(non_stop_sig, 250, 'flattop', scaling='spectrum')
    dt = 0.004  # Because 1 / 0.004 = 250
    Pxx, freqs = plt.psd(non_stop_sig, 256, 1 / dt, label='Non-Stop Signal PSD')
    'Plot Formatting'
    plt.legend()
    plt.xlim([12, 16])
    plt.ylim([-40, -5])
    plt.yticks(np.arange(-45, -5, step=10))
    plt.title(title)
    plt.grid(b=None)
    plt.show()
    return f, Pxx_spec


def sing_power(sig, plot, title):
    from scipy import signal
    # import matplotlib.mlab as mlab
    freqs, psd = signal.welch(sig)
    f, Pxx_spec = signal.periodogram(sig, 250, 'flattop', scaling='spectrum')
    if plot == 1:
        'Method 2'
        dt = 0.004  # Because 1 / 0.004 = 250
        Pxx, freqs = plt.psd(sig, 256, 1 / dt)
        plt.xlim([12, 16])
        title2 = title
        plt.title(title2)
        # plt.show()
        # Pxx, freqs = plt.psd(s, 512, 1 / dt)
    return f, Pxx_spec


def nancheck(data):
    'Check all channels for nan values.'
    'If even 1 nan found in an array, change all other values to nan.'
    'Assumes orientation Samples x Channels.'
    a, b = np.shape(data)
    # Preform zero array.
    nan_data = np.zeros((a, b))
    for i in range(b):
        if np.isnan(data[:, i]).any() is True:
            nan_data[:, i] = np.nan
            print('-------------------------NAN CHAN')
        if np.isnan(data[:, i]).any() is False:
            nan_data[:, i] = data[:, i]
            print('-------------------------NORM CHAN')
    return nan_data


def sing_data_extract(direc, ext, keyword, arr):
    'direc = get data directory.'
    'ext = select your file delimiter.'
    'keyword = unique filenaming word/phrase.'
    'arr = cycle in trial you want to extract.'

    'Example: '
    # eeg = pb.sing_data_extract('C:\P300_Project\Data_Aquisition\Data\\', '.npz', 'volt', 'arr_0')

    eegfiles = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    eeg_files = []
    for j in range(len(eegfiles)):
        if eegfiles[j].find(keyword) != -1:
            eeg_files.append(eegfiles[j])

    file_name = eeg_files[0]
    data = np.load(direc + file_name)
    data = data[arr]
    print('Extracted File: ', file_name, 'EEG Dims: ', np.shape(data))
    return data


def labels_inf_extract(direc, file_name):
    'direc - directory where label data is stored.'
    'file_name = name of labels file you want to extract info from.'

    'Example: '
    # matSize, num_seq, num_emojis, num_trials = pb.labels_inf_extract(
    #     'C:\P300_Project\Data_Aquisition\Data\Labels\\', '0001_trial_labels.npz')

    xA = np.load(direc + file_name)
    x1 = xA['arr_0']
    matSize = np.shape(x1)
    num_seq = matSize[0]
    num_emojis = matSize[1]
    num_trials = len(xA['arr_1'])
    return xA, matSize, num_seq, num_emojis, num_trials


def pathway_extract(direc, ext, keyword, full_ext):
    'direc = get data directory.'
    'ext = select your file delimiter.'
    'keyword = unique filenaming word/phrase.'

    'Example: '
    # eeg = pb.sing_data_extract('C:\P300_Project\Data_Aquisition\Data\\', '.npz', 'volt')

    files = [i for i in os.listdir(direc) if os.path.splitext(i)[1] == ext]
    _files = []
    for j in range(len(files)):
        if files[j].find(keyword) != -1:
            if full_ext != 1:
                _files.append(files[j])
            if full_ext == 1:
                _files.append(direc + files[j])
    return _files


def subject_extract(direc):
    'direc = get data directory with all subject data.'
    # sub_files = os.listdir(direc)
    sub_files = [i for i in os.listdir(direc) if i.find('0') == 0]
    eeg_files = []
    lab_files = []
    for i in range(len(sub_files)):
        eeg_files.append(direc + sub_files[i])
        lab_files.append(direc + sub_files[i] + '/Labels/')
    return eeg_files, lab_files


def path_subplant(main_direc, paths, lab):
    _files = []
    for i in range(len(paths)):
        if lab == 0:
            _files.append(main_direc + '/' + paths[i])
        elif lab == 1:
            _files.append(main_direc + '/Labels/' + paths[i])

    return _files


def time_stamper(data, show_stamps):
    # Isolate final channel from Cognionics array containing time-stamps.
    # ALso subtract 1st value in this array from all subsequent values.
    # This transforms data from system time to seconds units.
    'Example: '
    # eeg_time = pb.time_stamper(eeg, 1)

    eeg_time = np.abs(data[0, -1] - data[:, -1]) * 1000
    if show_stamps == 1:
        print('1st Time-Stamp: ', eeg_time[0], 'Last Time-Stamp: ', eeg_time[-1])
    return eeg_time


def simp_temp_axis(data, samp_ratekHz):
    'Generate an x axis for plotting time-series at diff Hz.'
    # Data is just your eeg array.
    # Samprate_kHZ needs to be given in KHz e.g. 0.5 = 500Hz.
    # Assumes Trials x Samples

    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    f = data.shape[0]

    constant = 1 / samp_ratekHz  # Temporal Constamt
    x = np.arange(0, (f * constant), constant)  # Temporal Axis

    return x


def temp_axis(data, samp_ratekHz, plt_secs):
    'Generate an x axis for plotting time-series at diff Hz.'
    # data is just your eeg array.
    # samprate needs to be given in KHz e.g. 0.5 = 500Hz.
    # plot_secs is the number of seconds you want plotting per trial.
    'Example: '
    # x = pb.temp_axis(eeg_avg, 0.5)
    f = data.shape
    f = np.amax(f)

    constant = 1 / samp_ratekHz  # Temporal Constamt
    x = np.arange(0, (f * constant), constant)  # Temporal Axis
    # Grab time-series to length of plot_secs.
    time_idx = [i for i, e in enumerate(x) if e == plt_secs]
    time_idx = time_idx[0]
    # Index array to size useing plot_secs.
    x = x[0:time_idx]

    return x, time_idx


def random_index(targCue, num_emoji):
    import random
    z = np.arange(num_emoji)
    random.shuffle(z)
    fin = z[0]

    print(fin)
    if fin == targCue:
        fin = z[1]
    return fin


def sess_inc(num_sess, sub_path, lab_path):
    sub_path2 = []
    lab_path2 = []
    for i in range(len(sub_path)):
        y = sub_path[i]
        if num_sess == 1:
            if y[31:36] == '00001':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
        if num_sess == 2:
            if y[31:36] == '00001' or y[31:36] == '00002':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
        if num_sess == 3:
            if y[31:36] == '00001' or y[31:36] == '00002' or y[31:36] == '00003':
                sub_path2 = np.append(sub_path2, sub_path[i])
                lab_path2 = np.append(lab_path2, lab_path[i])
    if num_sess == 4:
        sub_path2 = sub_path
        lab_path2 = lab_path
    return sub_path2, lab_path2


def band_power_plots(data, sing_plt, plotter):
    'Calculates bandpower values for 5 major EEG sub-bands.'
    'sing_plt = ON means it generates a single plt.'
    'plotter = changes whether present in absolute or relative power (out of 100).'
    fs = 500                                # Sampling rate (500 Hz)

    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)

    # Define EEG bands
    'Delta adapted to 0.5 | Ref: https://ieeexplore.ieee.org/abstract/document/5626721'
    eeg_bands = {'Delta': (0.5, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    # Plot.
    bands = np.arange(5)
    values = [eeg_band_fft['Delta'], eeg_band_fft['Theta'],
              eeg_band_fft['Alpha'], eeg_band_fft['Beta'], eeg_band_fft['Gamma']]
    sum_pow = np.sum(values)
    # Rel Power
    rel_pow = np.arange(len(values))
    for i in range(len(rel_pow)):
        rel_pow[i] = (values[i] / sum_pow) * 100

    if sing_plt == 'ON':
        fig, ax = plt.subplots()
        if plotter == 'ABS':
            plt.bar(bands, values)
        if plotter == 'REL':
            plt.bar(bands, rel_pow)
        plt.xticks(bands, ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'))
        plt.show()
    return values, rel_pow


def sub_band_chunk_plot(data, divs, pow_disp, plotter):
    'This creates a plot with 5 subplots, each providing a sub-band analysis of 1/nth of an experimental session.'
    'This should demonstrate the change in sub-band freqs, with more high amp low oscillations at the end.'
    'divs = number of divisions of data made across the experimental sessions.'
    'pow_dsip = either "ABS" for absolute values or "REL" for relative power.'
    'NOTE: No overlap is performed across chunks.'

    # Initializing data slicing for divisions.
    trials = data.shape[1]
    samp_ind = np.linspace(0, trials, divs + 1)
    samp_ind = np.round(samp_ind, decimals=0)
    samp_ind = samp_ind.astype('int')
    # X axis plotting labels.
    x_axis = ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')

    if divs > 1:
        if plotter == 'ON':
            fig, axes = plt.subplots(divs)
        for i in range(divs):
            eeg_chk = data[:, samp_ind[i]:samp_ind[i + 1]]
            # Avergae across division sub-samples (individual trials).
            eeg_chk = np.average(eeg_chk, axis=1)
            eeg_chk = np.expand_dims(eeg_chk, axis=1)
            values, rel_pow = band_power_plots(eeg_chk, sing_plt='OFF', plotter='OFF')
            if pow_disp == 'ABS':
                x_ax = autolabel(x_axis, values)
                if plotter == 'ON':
                    axes[i].bar(x_ax, values)
            elif pow_disp == 'REL':
                x_ax = autolabel(x_axis, rel_pow)
                if plotter == 'ON':
                    axes[i].bar(x_ax, rel_pow)
        if plotter == 'ON':
            fig.tight_layout()
            plt.show()
    else:
        eeg_chk = np.average(data, axis=1)
        eeg_chk = np.expand_dims(eeg_chk, axis=1)
        values, rel_pow = band_power_plots(eeg_chk, sing_plt='OFF', plotter='OFF')
        if pow_disp == 'ABS':
            x_ax = autolabel(x_axis, values)
            if plotter == 'ON':
                plt.bar(x_ax, values)
        elif pow_disp == 'REL':
            x_ax = autolabel(x_axis, values)
            if plotter == 'ON':
                plt.bar(x_ax, rel_pow)
        if plotter == 'ON':
            plt.show()
            plt.title('Sub-Band Plots')


def autolabel(x_axis, values):
    x = []
    for i in range(len(values)):
        xA = x_axis[i] + (': ') + np.array2string(np.array(values[i])) + ('%')
        if i == 0:
            x = xA
        else:
            x = np.append(x, xA)
    return x


def lda_loc_extract(dat_direc, verbose, norm, num_trials):
    'Extracts data from Localizer experiments for LDA analysis.'
    'dat_direc = location of data files.'
    'num_trials = number trials you want to extract from the experimental session, if [] it takes all trials.'
    '**kwargs = is used to detect if num_trials has been specifed. '
    'verbose = if 1 it prints dim info on returned variables.'
    'norm = add normalization if == 1.'
    # Data Pathway Extraction.
    lab_direc = dat_direc + 'Labels/'
    dat_files = pathway_extract(dat_direc, '.npy', 'Volt', full_ext=1)
    lab_files = pathway_extract(lab_direc, '.npz', 'Labels', full_ext=1)
    labels = np.load(lab_files[0])['arr_0']
    target_names = np.array(['P300', 'NP300'])
    '_________P300 Averaging_________'
    np3 = []
    p3 = []
    # Iterator Variables.
    iterator = num_trials
    if num_trials > len(labels):
        iterator = len(labels)
    labels = labels[0:iterator]
    # Extrsaction.
    for i in range(iterator):
        'Pre-Process Data Chunk'
        eeg = np.load(dat_files[i])
        # For Cz / Target electrodes breakdown.
        eeg = prepro(eeg, samp_ratekHz=0.5, zero='ON', ext='INC-Ref', elec=[0, 1, 3, 4],
                     filtH='ON', hlevel=1, filtL='ON', llevel=10,
                     notc='LOW', notfq=50, ref_ind=-7, ref='A2', avg='ON')
        # NP300 Trials.
        if labels[i] == 0:
            if np3 == []:
                np3 = eeg
                np3 = np.expand_dims(np3, axis=2)
            else:
                eeg = np.expand_dims(eeg, axis=2)
                np3 = np.append(np3, eeg, axis=2)
        # P300 Trials.
        if labels[i] == 1:
            if p3 == []:
                p3 = eeg
                p3 = np.expand_dims(p3, axis=2)
            else:
                eeg = np.expand_dims(eeg, axis=2)
                p3 = np.append(p3, eeg, axis=2)
    # Conjoin.
    p3 = np.squeeze(p3)
    np3 = np.squeeze(np3)
    eeg = np.append(p3, np3, axis=1)
    if norm == 1:
        eeg = scaler1D(eeg, -1, 1)
    eeg = np.swapaxes(eeg, 0, 1)
    # Info
    if verbose == 1:
        print('X DIMS: ', eeg.shape, '\n', eeg[50:54, 0:4])
        print('y DIMS: ', labels.shape, labels[0:4])
        print('Names DIMS: ', target_names.shape, target_names[0:4])
    return eeg, labels, target_names
