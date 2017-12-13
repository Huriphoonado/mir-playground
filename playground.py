# Willie Payne
# Run Command: python3 playground.py

# ----------------- Imports
from multiprocessing import Pool  # For parallel processing

import json  # For exporting predictions

import numpy as np  # For numerous calculations

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import librosa
import librosa.display

import medleydb as mdb

from mir_eval import melody as mel_eval

# ----------------- Global Variables
target_sr = 22050  # Lower this if you decide to downsample
original_sr = 44100
n_fft = 1024
win_length = 1024
hop_length = int(256 * (target_sr / original_sr))  # So time points line up
window = 'hann'
num_processes = 4  # Number of parallel processes for supported code blocks
global_mode = 'all'  # ['voicing' | 'melody' | 'all']


# ----------------- Functions
def generate_train_and_test_data(size, test_mode=False):
    '''
        Creates all of the training and test data that we will need
        Inputs:
            Float ranging from 0 to 1 referring to how to split data up
            Boolean when set to True will only run on 3 audio files
        Outputs:
            train: List of multitrack objects to be used for training
            test: List of multitrack objects to be used for testing
    '''
    if test_mode:
        train = [mdb.MultiTrack('MusicDelta_Reggae'),
                 mdb.MultiTrack('MusicDelta_Rockabilly')]
        test = [mdb.MultiTrack('MusicDelta_Shadows')]
        return train, test

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

    with Pool(processes=num_processes) as pool:
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
                    times: list containing times for all annotations
    '''
    t_id = mtrack.track_id
    y, sr = librosa.load(mtrack.mix_path, res_type='kaiser_fast',
                         sr=target_sr, mono=True)  # or 'kaiser_best'

    # Each annotation contains time stamp, and pitch value
    times, annotation = zip(*mtrack.melody2_annotation)
    annotation = list(annotation)
    times = list(times)

    normalized = final_steps(n_func(y))

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

    if global_mode == 'all':
        annotation = hz_to_note_zeros(annotation)
    elif global_mode == 'voicing':
        annotation = np.array([int(bool(v)) for v in annotation])
    else:  # If just melody, this will take a little bit more work
        annotation = hz_to_note_zeros(annotation)

    # annotation = hz_to_note_zeros(annotation)

    print('Normalized', t_id, 'with', len(normalized), 'feature vectors')
    return {'t_id': t_id, 'features': normalized,
            'labels': annotation, 'times': times}


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


def with_cube_root(y):
    return np.cbrt(with_stft(y))


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
def train_model_svm(train_features, train_labels):
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


def train_model_forest(train_features, train_labels):
    clf = RandomForestClassifier(max_depth=2, random_state=0,
                                 n_jobs=num_processes)
    clf.fit(train_features, train_labels)
    print(clf.n_classes_)
    return clf


def predict(clf, all_test_data):
    '''
        Run predictions on all tracks
        Inputs:
            Classifier created by train_model function
            List of dicts containing
        Output: Modified list of dicts containing 'guesses' field
    '''
    for track in all_test_data:
        track['guesses'] = clf.predict(track['features'])

    return all_test_data


def export_predictions(all_test_data):
    '''
        Exports the predictions dict into a json file so that results
        be loaded and graphed in another program
        Inputs: List of dicts
    '''
    # Convert frequency back to hertz, and then other things to python lists
    copy = []
    for d in all_test_data:
        new_d = {}
        if global_mode != 'voicing':  # Convert Note Names to Frequencies
            new_d['labels'] = (note_to_hz_zeros(d['labels'])).tolist()
            new_d['guesses'] = note_to_hz_zeros(d['guesses']).tolist()
        else:
            new_d['labels'] = d['labels'].tolist()
            new_d['guesses'] = d['guesses'].tolist()

        new_d['t_id'] = d['t_id']
        copy.append(new_d)

    with open('predict.json', 'w') as file:
        json.dump(copy, file)


def evaluate_model_voicing(test_guesses, test_labels):
    '''
        If we are only looking at voicing, then we only care about some metrics
        Inputs:
            1d Boolean np.array containing all predictions made by the model
            1d Boolean np.array containing all ground truth labels
    '''
    ref_voicing = test_labels.astype(bool)
    est_voicing = test_guesses.astype(bool)

    print('Evaluating voicing...')
    vx_recall, vx_false_alarm = mel_eval.voicing_measures(ref_voicing,
                                                          est_voicing)
    print('Evaluating overall accuracy...')
    overall_accuracy = mel_eval.overall_accuracy(ref_voicing, test_labels,
                                                 est_voicing, test_guesses,
                                                 cent_tolerance=50)

    metrics = {
        'vx_recall': vx_recall,
        'vx_false_alarm ': vx_false_alarm,
        'overall_accuracy': overall_accuracy
    }

    for m, v in metrics.items():  # Python2 is iteritems I think
        print(m, ':', v)
    return metrics


def evaluate_model_all(test_guesses, test_labels):
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
    print('Running conversions...')
    ref_freq = note_to_hz_zeros(test_labels)  # And back to Hz!
    est_freq = note_to_hz_zeros(test_guesses)

    ref_cent = mel_eval.hz2cents(ref_freq)  # Then to cents...
    est_cent = mel_eval.hz2cents(est_freq)

    ref_voicing = mel_eval.freq_to_voicing(ref_freq)[1]  # And voicings!
    est_voicing = mel_eval.freq_to_voicing(est_freq)[1]

    print('Evaluating voicing...')
    vx_recall, vx_false_alarm = mel_eval.voicing_measures(ref_voicing,
                                                          est_voicing)

    print('Evaluating pitch...')
    raw_pitch = mel_eval.raw_pitch_accuracy(ref_voicing, ref_cent,
                                            est_voicing, est_cent,
                                            cent_tolerance=50)

    print('Evaluating chroma...')
    raw_chroma = mel_eval.raw_chroma_accuracy(ref_voicing, ref_cent,
                                              est_voicing, est_cent,
                                              cent_tolerance=50)

    print('Evaluating overall accuracy...')
    overall_accuracy = mel_eval.overall_accuracy(ref_voicing, ref_cent,
                                                 est_voicing, est_cent,
                                                 cent_tolerance=50)

    metrics = {
        'vx_recall': vx_recall,
        'vx_false_alarm ': vx_false_alarm,
        'raw_pitch': raw_pitch,
        'raw_chroma': raw_chroma,
        'overall_accuracy': overall_accuracy
    }

    for m, v in metrics.items():  # Python2 is iteritems I think
        print(m, ':', v)

    return metrics


# ----------------- Main Function
def main():
    n_func = with_stft  # Choose your weapon!
    train, test = generate_train_and_test_data(0.15, test_mode=True)

    print('Extracting Training Features..........')
    all_training_data = normalize_all(n_func, train)
    print('Extracting Training Features..........Done')

    print('Training  Model..........')
    train_features = concat(all_training_data, 'features')
    train_labels = concat(all_training_data, 'labels')
    clf = train_model_forest(train_features, train_labels)
    print('Training  Model..........Done')

    print('Extracting Test Features..........')
    all_test_data = normalize_all(n_func, test)
    print('Extracting Test Features..........Done')

    print('Making Predictions..........', end='')
    predictions = predict(clf, all_test_data)
    print('Done')

    print('Exporting Predictions..........', end='')
    export_predictions(all_test_data)
    print('Done')

    print('Evaluating Results..........')
    test_guesses = concat(predictions, 'guesses')
    test_labels = concat(predictions, 'labels')
    if global_mode == 'all':
        evaluate_model_all(test_guesses, test_labels)
    elif global_mode == 'voicing':
        evaluate_model_voicing(test_guesses, test_labels)
    print('Evaluating Results..........Done')


if __name__ == '__main__':
    main()
