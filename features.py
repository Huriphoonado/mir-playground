# Possible Feature Representations to convert input audio

# ----------------- Imports
import numpy as np
import librosa

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
    return final_steps(np.cbrt(with_stft(y)))


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
def transform(type):
    transformations = {
        'stft': with_stft,
        'cube_root': with_cube_root
    }

    return transformations[type]
