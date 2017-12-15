# Possible Feature Representations to convert input audio

# TODO - Figure out how to make final_steps a higher order function to
# reduce code duplication for functions below that have the same steps (stft)

# ----------------- Imports
import numpy as np
import librosa
import scipy

# ----------------- Global Variables
target_sr = 22050  # Lower this if you decide to downsample
original_sr = 44100  # Sampling rate for tracks in Medleydb
n_fft = 1024
win_length = 1024
hop_length = int(256 * (target_sr / original_sr))  # So time points line up
window = 'hann'


# ----------------- Transformation Functions
def with_stft(y):
    s_array = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)

    abs_s = np.absolute(s_array)  # converts complex64 values to floats
    return final_steps(abs_s)


def with_cube_root(y):
    s_array = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)

    abs_s = np.absolute(s_array)
    cbrt = np.cbrt(abs_s)
    return final_steps(cbrt)


def stft_no_loss(y):
    stft = np.zeros((win_length, int(np.ceil(len(y)/hop_length)+1)))
    stft = stft+0j

    y = np.pad(y, int(n_fft / 2), mode='reflect')
    stft_buffer = librosa.util.frame(y, frame_length=win_length,
                                     hop_length=hop_length)

    for i in range(stft_buffer.shape[1]):
        stft[:, i] = scipy.fftpack.fft(stft_buffer[:, i], n_fft)

    print(stft.shape)
    return stft


def with_autocorrelation(y):
    min_lag = 15
    max_lag = 800

    N_l = np.array(np.linspace((win_length)-min_lag,
                   win_length-max_lag, max_lag-min_lag+1), ndmin=2)
    N_l = np.transpose(N_l)
    N_l = 1 / N_l

    stft = stft_no_loss(y)

    acf = np.zeros(stft.shape)
    acf = acf + 0j
    for i in range(stft.shape[1]):
        acf[:, i] = scipy.fftpack.ifft(np.power(np.absolute(stft[:, i]), 2))

    acf = np.real(acf)

    acf = acf[min_lag:max_lag+1, :]

    acf = N_l*acf

    return final_steps(acf)


# TODO - Fix divide by zero error, occurs when taking the log of 0
def with_cepstrum(y):
    stft = stft_no_loss(y)

    cepstrum = np.zeros(stft.shape)
    cepstrum = cepstrum + 0j

    for i in range(stft.shape[1]):
        cepstrum[:, i] = scipy.fftpack.ifft(np.log10(np.absolute(stft[:, i])))

    cepstrum = np.real(cepstrum)
    return final_steps(cepstrum)


def with_salience(y):
    s_array = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)
    s_array = np.abs(s_array)
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)
    h_range = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    salience = librosa.salience(s_array, freqs, h_range)

    return final_steps(salience)


def final_steps(s):
    '''
    Wraps around normalize function and runs any final steps common
    to all functions to finish processing a feature vector array
    Input: 2d np.array
    Output: Modified 2d np.array with range 0-1 and axes swapped
    '''
    # Normalize entire matrix from 0 to 1 - useful since amplitude differs
    min_val = np.amin(s)
    s = s - min_val
    max_val = np.amax(s)
    s = s / max_val

    # Swap axes so that dimensions line up with annotation
    return np.swapaxes(s, 0, 1)


# ----------------- Function Generator
def generate_transform(type):
    '''
        Returns the right transform function based on the string inputted
        If the string 'options' is inputted, it will return a dict containing
        modes rather than a function
        Input: String containing ['options' | 'voicing' | 'melody' | 'all']
        Output: Either:
            Transform function corresponding to input
            Dict containing possible transformation functions for user input
    '''
    transformations = {
        'stft': with_stft,
        'cube_root': with_cube_root,
        'autocorr': with_autocorrelation,
        'salience': with_salience
    }

    if type == 'options':
        return {i: k for i, k in enumerate(transformations)}
    else:
        return transformations[type]
