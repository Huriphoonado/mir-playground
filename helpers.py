# Useful Helper Functions that get called throughout all of our code
# This includes data conversion, array concatenation, etc...

# ----------------- Imports
import numpy as np
import librosa
from collections import Counter


# ----------------- Helper Functions
def hz_to_note_zeros(annotation):
    '''
        Special function so that zeros represent silence
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
        Special function so that zeros represent silence
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


def midi_to_hz_zeros(annotation):
    '''
        Special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing frequencies instead of note names
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = 0
        if a != 0:
            new_a = librosa.midi_to_hz(a)
        new_values = np.append(new_values, new_a)

    return new_values


def hz_to_midi_zeros(annotation):
    '''
        Special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing frequencies instead of note names
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = 0
        if a != 0:
            new_a = librosa.hz_to_midi(a)
        new_values = np.append(new_values, new_a)

    return new_values


def note_to_midi_zeros(annotation):
    '''
        Special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing frequencies instead of note names
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = 0
        if a != '0':
            new_a = librosa.note_to_midi(a)
        new_values = np.append(new_values, new_a)

    return new_values


def midi_to_note_zeros(annotation):
    '''
        Special function so that zeros represent silence
        Input: Annotation List taken straight from mtrack
        Output: 1d np.array containing frequencies instead of note names
    '''
    new_values = np.array([])

    for a in annotation:
        new_a = '0'
        if a != 0:
            new_a = librosa.midi_to_note(a)
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


def count_pitches(annotation):
    '''
        Counts the unique classes in an annotation
        Input: 1d np.array of either note or voicing annotations
        Output: dict where class is the key and count is the value
    '''
    unique, counts = np.unique(annotation, return_counts=True)
    pairs = np.asarray((unique, counts)).T  # 2d np.array
    string_dict = dict(pairs.tolist())  # Converts counts to strings :(
    int_dict = {k: int(v) for k, v in string_dict.items()}

    return int_dict


def common_pitches(data, threshold):
    '''
        Aggregates unique pitches across the entire inputted dataset
        Inputs:
            Dataset containing 'class_counts' field
            Threshold value for to only include pitches above the line
        Outputs:
            Dict containing total counts based on each pitch
            List containing all the pitches we plan to remove
    '''
    counts_list = [d['class_counts'] for d in data]
    counter = Counter()

    for d in counts_list:
        counter.update(d)

    all_counts = dict(counter)
    to_remove = {k: v for k, v in counter.items() if v < threshold or k == '0'}

    return to_remove, all_counts


def keep_some_frames(track_dict, to_remove):
    '''
        Given a list of labels we do not want to train on, this function will
        update the labels, features, and times lists to remove those frames
        Inputs:
            An audio track dict to be altered
            A dict/list containing labels to wish to remove
        Output: Modified dict
    '''

    lbls = track_dict['labels']
    times = track_dict['times']
    features = track_dict['features']

    # Create a list of indices from labels to filter out
    i_to_keep = [i for i, lbl in enumerate(lbls) if lbl not in to_remove]

    # Then filter those values from times, labels, features
    track_dict['labels'] = np.array([lbls[i] for i in i_to_keep])
    track_dict['times'] = [times[i] for i in i_to_keep]  # Regular old list
    track_dict['features'] = np.array([features[i] for i in i_to_keep])

    return track_dict


def input_string(prompt_type, options_dict):
    '''
        Creates a string for user input
    '''
    i_string = 'Please choose a(n) ' + prompt_type + ' mode: \n'
    for k, v in options_dict.items():
        i_string += str(k) + ': ' + v + '\n'
    i_string += 'Your (integer) choice: '

    return i_string
