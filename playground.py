# Willie Payne
# Run Command: python3 playground.py

# ----------------- Imports
from __future__ import print_function

import numpy as np

from sklearn import svm

# import matplotlib.pyplot as plt
# import matplotlib.style as ms

import librosa
import librosa.display

import medleydb as mdb

# import mir_eval

# ms.use('seaborn-muted')


# ----------------- Global Variables
target_sr = 22050  # Lower this if you decide to downsample
original_sr = 44100
n_fft = 1024
win_length = 1024
hop_length = int(256 * (target_sr / original_sr))  # So time points line up
window = 'hann'


# ----------------- Functions
def generate_train_and_test_data(size):
    '''
        Creates all of the training and test data that we will need
        Inputs: Float ranging from 0 to 1 referring to how to split data up
        Outputs:
            train: List of multitrack objects to be used for training
            test: List of multitrack objects to be used for testing
    '''
    generator = mdb.load_melody_multitracks()
    melody_ids = [mtrack.track_id for mtrack in generator]
    splits = mdb.utils.artist_conditional_split(trackid_list=melody_ids,
                                                test_size=size, num_splits=1)

    train = [mdb.MultiTrack(t_id) for t_id in splits[0]['train']]
    test = [mdb.MultiTrack(t_id) for t_id in splits[0]['test']]
    return train, test


def normalize_all(n_func, train_or_test):
    '''
        Iterates through all provided data and normalizes each mtrack
        Inputs:
            Function referring to the normalization function to run
            List containing either training data or test data
        Outputs:
            List containing feature vectors from all audio files
            List containing labels/annotations from all audio files
    '''
    all_features = []
    all_labels = []

    for mtrack in train_or_test:
        y, sr = librosa.load(mtrack.mix_path, res_type='kaiser_fast',
                             sr=target_sr, mono=True)  # or 'kaiser_best'
        t_id = mtrack.track_id

        # Each annotation contains time stamp, and pitch value
        time, annotation = zip(*mtrack.melody2_annotation)
        annotation = list(annotation)

        normalized = normalizer(n_func, y)

        # Make sure that feature vector is the same length as the labels!
        # Question maybe for Rachel? If difference is only 1??
        if len(normalized) != len(annotation):
            if len(normalized) - len(annotation) == 1:
                normalized = normalized[:-1]  # remove extra vector from end
            else:  # Something really went wrong otherwise!
                print('Error! Feature vector differs in length from labels.')
                print(t_id, 'labels has size:', len(annotation))
                print(t_id, 'features has size:', len(normalized))
                quit()

        all_features.append(normalized)
        all_labels.append(annotation)
        print('Normalized', t_id, 'with', len(normalized), 'feature vectors')

    return all_features, all_labels


def normalizer(n_func, audio):
    s = n_func(audio)  # Call chosen normalize function

    # We'll need to change the shape so that it is t lists of size n_fft
    # rather than n_fft lists of size t
    s_t = np.swapaxes(s, 0, 1)

    # Then convert to a python list so it is the same type as annotations
    s_ts = s_t.tolist()
    return s_ts


def with_stft(y):
    s_array = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window)

    abs_s = np.absolute(s_array)  # converts complex64 values to floats
    return abs_s


def with_51_pt_norm(y):
    s = with_stft(y)
    # Magnitude of the STFT is normalized within each time frame to achieve
    # zero mean and unit variance over a 51-frame local frequency window
    return s


def with_cube_root(y):
    return np.cbrt(with_stft(y))


def train_model(train_features, train_labels):
    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    return clf


def predict(clf, test_features):
    return clf.predict(test_features)


def evaluate_model(predictions, test_labels):
    # http://craffel.github.io/mir_eval/#module-mir_eval.melody

    # Required by functions below - Maybe call before svm??
    # mir_eval.melody.to_cent_voicing()

    # Evaluation Metrics
    # mir_eval.melody.freq_to_voicing(frequencies)
    # mir_eval.melody.voicing_measures(ref_voicing, est_voicing)

    # mir_eval.melody.raw_pitch_accuracy()
    # mir_eval.melody.raw_chroma_accuracy() # Would require extra step
    # mir_eval.melody.overall_accuracy()

    # Probably want to write to a text file
    pass


# ----------------- Main Function
def main():
    train, test = generate_train_and_test_data(0.15)  # multiple splits??

    print('Extracting Training Features..........')
    train_features, train_labels = normalize_all(with_stft, train)
    print('Extracting Training Features..........Done')

    print('Training  Model..........', end='')
    clf = train_model(train_features, train_labels)
    print('Done')

    print('Extracting Test Features..........')
    test_features, test_labels = normalize_all(with_stft, test)
    print('Extracting Test Features..........Done')

    print('Making Predictions..........', end='')
    predictions = predict(clf, test_features)
    print('Done')

    print('Evaluating Results..........', end='')
    evaluate_model(predictions, test_labels)
    print('Done')


if __name__ == '__main__':
    main()


# ----------------- Steps
# Audio Sample combined to 1-channel
# Downsampled to 8 kHz
# Converted to STFT
#   N = 1024 with a Hanning Window
#   N Overlap = 944
# Use as SVM - N-way multi-class discrimination

# Normalization Types Used
#   STFT
#       1. Normalized such that the maximum energy frame in each song
#           has a value equal to 1
#       2. 51 point Norm - Magnitude normalized within each time frame to
#           achieve zero mean and unit variance
#       3. Cube root compression applied to make larger magnitudes appear more
#           similar
#   MEL
#       4. Autocorrelation
#       5. Cepstrum
#       6. Normalized Autocorrelation by local mean and variance equalization
#           as applied to the spectra
#       7. Liftering Cepstrum - Scaling higher order cepstra by exponential
#           weight

# Use either 1 or 2 for annotations
# Use mdb.utils.artist_conditional_split for test and training data
# 256 samples at a sample rate of 44100, ~5.8 milliseconds
#   This is probably our hop size? Turns out to be hop size * 2?
# We may only wish to keep values that actually have melody which would add an
# extra step?

# Seemed to freeze when reading
# whatever comes after MichaelKropf_AllGoodThings
