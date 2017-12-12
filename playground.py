# Willie Payne
# Run Command: python3 playground.py

# ----------------- Imports
from __future__ import print_function

from multiprocessing import Pool

import numpy as np

from sklearn import svm

# import matplotlib.pyplot as plt
# import matplotlib.style as ms

import librosa
import librosa.display

import medleydb as mdb

from mir_eval import melody as mel_eval

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
        Outputs: List of dicts
    '''
    func_with_data = [(n_func, mtrack) for mtrack in train_or_test]

    with Pool(processes=3) as pool:
        feature_dict = pool.starmap(load_and_normalize, func_with_data)

    return feature_dict


# Post processing - Subtract min and divide by max
def load_and_normalize(n_func, mtrack):
    '''
        Loads the selected audio file and runs normalization on it
        Inputs:
            Normalization function to use (begins with 'with_')
            mtrack object from Medleydb
        Output: Dict containing
                    t_id: track id
                    features: 2d np.array containing normalized feature vector
                    labels: 1d np.array containing all labels
                    times: 1d np.array containing times for all annotations
    '''
    t_id = mtrack.track_id
    y, sr = librosa.load(mtrack.mix_path, res_type='kaiser_fast',
                         sr=target_sr, mono=True)  # or 'kaiser_best'

    # Each annotation contains time stamp, and pitch value
    times, annotation = zip(*mtrack.melody2_annotation)
    annotation = list(annotation)
    times = list(times)

    normalized = normalizer(n_func, y)

    if len(normalized) != len(annotation):
        if len(normalized) - len(annotation) == 1:
            normalized = normalized[:-1]  # remove extra vector from end
        elif len(annotation) - len(normalized) == 1:
            annotation = annotation[:-1]
            times = times[:-1]
        else:  # Something really went wrong otherwise!
            print('Error! Feature vector differs in length from labels.')
            print(t_id, 'labels has size:', len(annotation))
            print(t_id, 'features has size:', len(normalized))
            quit()

    annotation = hz_to_note_zeros(annotation)

    print('Normalized', t_id, 'with', len(normalized), 'feature vectors')
    return {'t_id': t_id, 'features': normalized,
            'labels': annotation, 'times': np.array(times)}


def normalizer(n_func, audio):
    s = n_func(audio)  # Call chosen normalize function

    # We'll need to change the shape so that it is t lists of size n_fft/2 + 1
    # rather than n_fft/2 + 1 lists of size t
    s_t = np.swapaxes(s, 0, 1)

    # Then convert to a python list so it is the same type as annotations
    # s_ts = s_t.tolist()  # This causes everything to break!!
    return s_t


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


# Use Salience Function - TimeFreq
# Use cqt
# Somewhere use HzToDb

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


# Do quarter tone MIDI values
def train_model(train_features, train_labels):
    '''
        Uses an SVM to train the melody prediction model
        Inputs:
            2d np.array containing all feature vectors for each time
            1d np.array containing labels for all feature vectors*
            * The length of both lists must be equal
        Output: A classifier to be used for melody prediction

    '''
    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    return clf


def predict(clf, all_test_data):
    '''
        Run predictions on all tracks
        Inputs:
            Classifier created by train_model function
            Test Data Set

    '''
    for track in all_test_data:  # Probably could parallelize!
        track['guesses'] = clf.predict(track['features'])

    return all_test_data


def evaluate_model(test_guesses, test_labels):
    '''
        Run standard Mirex evaluations on all test data
        Converts to cents and voicing following mirex guidlines:
            http://craffel.github.io/mir_eval/#module-mir_eval.melody
        Inputs:
            1d np.array containing all predictions made by the model
            1d np.array containing all ground truth labels
        Outputs:
            Dict holding results of all evaluations
    '''
    ref_freq = note_to_hz_zeros(test_labels)
    est_freq = note_to_hz_zeros(test_guesses)

    ref_cent = mel_eval.hz2cents(ref_freq)
    est_cent = mel_eval.hz2cents(est_freq)

    ref_voicing = mel_eval.freq_to_voicing(ref_freq)
    est_voicing = mel_eval.freq_to_voicing(est_freq)

    vx_recall, vx_false_alarm = mel_eval.voicing_measures(ref_voicing,
                                                          est_voicing)

    raw_pitch = mel_eval.raw_pitch_accuracy(ref_voicing, ref_cent,
                                            est_voicing, est_cent,
                                            cent_tolerance=50)

    raw_chroma = mel_eval.raw_chroma_accuracy(ref_voicing, ref_cent,
                                              est_voicing, est_cent,
                                              cent_tolerance=50)

    overall_accuracy = mel_eval.overall_accuracy(ref_voicing, ref_cent,
                                                 est_voicing, est_cent,
                                                 cent_tolerance=50)

    metrics = {
        'vx_recall': vx_recall,
        'vx_false_alarm ': vx_false_alarm,
        'raw_pitch': raw_pitch,
        'raw_chroma': raw_chroma,
        'overall_accuracy': overall_accuracy,
    }

    for m, v in metrics.items():  # Python2 is iteritems I think
        print(m, ':', v)

    return metrics


# ----------------- Main Function
def main():
    n_func = with_stft
    train, test = generate_train_and_test_data(0.90)  # multiple splits??

    print('Extracting Training Features..........')
    all_training_data = normalize_all(n_func, train)
    print('Extracting Training Features..........Done')

    print('Training  Model..........')
    train_features = concat(all_training_data, 'features')
    train_labels = concat(all_training_data, 'labels')
    clf = train_model(train_features, train_labels)
    print('Training  Model..........Done')

    print('Extracting Test Features..........')
    all_test_data = normalize_all(n_func, test)
    print('Extracting Test Features..........Done')

    print('Making Predictions..........', end='')
    predictions = predict(clf, all_test_data)
    print('Done')

    print('Evaluating Results..........', end='')
    test_guesses = concat(predictions, 'guesses')
    test_labels = concat(predictions, 'labels')
    evaluate_model(test_guesses, test_labels)
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
