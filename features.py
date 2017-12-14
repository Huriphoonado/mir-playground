# Possible Feature Representations to convert input audio

# TODO - Figure out how to make final_steps a higher order function to
# reduce code duplication for functions below that have the same steps (stft)

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
    s_array = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)

    abs_s = np.absolute(s_array)
    cbrt = np.cbrt(abs_s)
    return final_steps(cbrt)


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
        'cube_root': with_cube_root
    }

    if type == 'options':
        return {i: k for i, k in enumerate(transformations)}
    else:
        return transformations[type]
