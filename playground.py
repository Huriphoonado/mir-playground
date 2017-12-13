# Willie Payne
# Run Command: python3 playground.py

# ----------------- Imports
from multiprocessing import Pool  # For parallel processing

import numpy as np  # For numerous calculations

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import librosa
import librosa.display

import medleydb as mdb

# Functions that we have written
import features
import helpers as hr
import exporter
import evaluate


# ----------------- Global Variables
num_processes = 4  # Number of parallel processes for supported code blocks
global_mode = 'voicing'  # ['voicing' | 'melody' | 'all']


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
                         sr=features.target_sr, mono=True)  # or 'kaiser_best'

    # Each annotation contains time stamp, and pitch value
    times, annotation = zip(*mtrack.melody2_annotation)
    annotation = list(annotation)
    times = list(times)

    normalized = n_func(y)

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
        annotation = hr.hz_to_note_zeros(annotation)
    elif global_mode == 'voicing':
        annotation = np.array([int(bool(v)) for v in annotation])
    else:  # If just melody, this will take a little bit more work
        annotation = hr.hz_to_note_zeros(annotation)

    # annotation = hr.hz_to_note_zeros(annotation)

    print('Normalized', t_id, 'with', len(normalized), 'feature vectors')
    return {'t_id': t_id, 'features': normalized,
            'labels': annotation, 'times': times}


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
    clf = RandomForestClassifier(max_depth=4, random_state=0,
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


# ----------------- Main Function
def main():
    n_func = features.transform('stft')  # Choose your weapon!
    train, test = generate_train_and_test_data(0.15, test_mode=True)

    print('Extracting Training Features..........')
    all_training_data = normalize_all(n_func, train)
    print('Extracting Training Features..........Done')

    print('Training  Model..........')
    train_features = hr.concat(all_training_data, 'features')
    train_labels = hr.concat(all_training_data, 'labels')
    clf = train_model_forest(train_features, train_labels)
    print('Training  Model..........Done')

    print('Extracting Test Features..........')
    all_test_data = normalize_all(n_func, test)
    print('Extracting Test Features..........Done')

    print('Making Predictions..........', end='')
    predictions = predict(clf, all_test_data)
    print('Done')

    print('Exporting Predictions..........', end='')
    exporter.predictions(all_test_data, global_mode)
    print('Done')

    print('Evaluating Results..........')
    test_guesses = hr.concat(predictions, 'guesses')
    test_labels = hr.concat(predictions, 'labels')
    ev = evaluate.generate_eval(global_mode)
    ev(test_guesses, test_labels)
    print('Evaluating Results..........Done')


if __name__ == '__main__':
    main()
