# Useful Helper Functions that get called throughout all of our code
# This includes data conversion, array concatenation, etc...

# ----------------- Imports
import numpy as np
import librosa


# ----------------- Helper Functions
def hz_to_note_zeros(annotation):
    '''
        We need a special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing note names instead of frequencies
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = '0'
        if a != 0:
            new_a = librosa.hz_to_note(a, cents=False)
        new_values = np.append(new_values, new_a)

    return new_values


def note_to_hz_zeros(annotation):
    '''
        We need a special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing frequencies instead of note names
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = 0
        if a != '0':
            new_a = librosa.note_to_hz(a)
        new_values = np.append(new_values, new_a)

    return new_values


def concat(data, feature_type):
    '''
    Concatenates all track information into one np.array to be used for model
    Inputs:
        List of dicts containing all song data
        String representing the feature type to concatenate
    Output: 1d or 2d np.array
    '''
    all_data = False

    for d in data:
        if type(all_data) is bool:  # Array dimensions set with first vector
            all_data = d[feature_type]
        else:
            all_data = np.concatenate((all_data, d[feature_type]), axis=0)

    print(feature_type, 'array has shape: ', all_data.shape)
    return all_data
