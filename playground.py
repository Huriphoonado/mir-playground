# Willie Payne
# Run Command: python3 playground.py

# TODO
# Implement CLI
#   List Feature for generator functions
# Add other functions to features.py and test
# Implement
# Check if voicing overall evaluation is actually accurate

# ----------------- Imports
from multiprocessing import Pool  # For parallel processing

import numpy as np  # For numerous calculations

from sklearn import svm  # Machine learning algorithms
from sklearn.ensemble import RandomForestClassifier

import librosa  # Audio processing and data conversions

# Functions that we have written
import split  # For splitting train/validation/test
import features  # Feature transformation methods
import helpers as hr  # Useful helper functions
import exporter  # For exporting data to JSON
import evaluate  # Final step evaluation functions


# ----------------- Global Variables
num_processes = 4  # Number of parallel processes for supported code blocks
e_mode = 'voicing'  # Evaluation mode
n_mode = 'stft'  # Type of Feature transformation to use
s_mode = 'quick'  # Type of train/test/validate split to use


# ----------------- Functions
def get_started():
    global e_mode  # Only time global keyword is used since these
    global n_mode
    global s_mode

    e_options = evaluate.generate_eval('options')
    n_options = features.generate_transform('options')
    s_options = split.generate_split('options')

    e_choice = input(hr.input_string('evaluation', e_options))
    n_choice = input(hr.input_string('feature', n_options))
    s_choice = input(hr.input_string('split', s_options))

    try:
        e_mode = e_options.get(int(e_choice), 'voicing')
        n_mode = n_options.get(int(n_choice), 'stft')
        s_mode = s_options.get(int(s_choice), 'quick')
    except:
        print('Oops, you must have typed something weird. Try running again.')
        quit()


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

    normalized = n_func(y)  # Transform to feature representation

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

    if e_mode == 'voicing':
        annotation = np.array([int(bool(v)) for v in annotation])
    else:
        annotation = hr.hz_to_note_zeros(annotation)

    # count unique pitches/voicings
    class_counts = hr.count_pitches(annotation)

    print('Normalized', t_id, 'with', len(normalized), 'feature vectors')
    return {'t_id': t_id, 'features': normalized,
            'labels': annotation, 'times': times,
            'class_counts': class_counts}


def only_voiced_frames(data, to_remove=['0']):
    '''
        Mainly used by the melody mode classification to hold voiced frames
        Inputs:
            List of dicts containing all train or test data
            List of pitches too rare to classify in addition to 0
        Outputs:
            Modified data list where undesirable frames/times/classes are
            cut out
    '''
    arg_list = [(track, to_remove) for track in data]
    with Pool(processes=num_processes) as pool:
        updated_tracks = pool.starmap(hr.keep_some_frames, arg_list)

    return updated_tracks


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
    get_started()  # Choose your weapon!
    e_func = evaluate.generate_eval(e_mode)
    n_func = features.generate_transform(n_mode)
    s_func = split.generate_split(s_mode)

    print('Splitting Train and Test Sets..........', end='')
    train, test = s_func()
    print('Done')

    print('Extracting Training Features..........')
    all_training_data = normalize_all(n_func, train)
    to_remove, train_counts = hr.common_pitches(all_training_data, 50)
    print('Extracting Training Features..........Done')

    if e_mode == 'melody':
        print('Removing Unvoiced Frames From Train..........', end='')
        all_training_data = only_voiced_frames(all_training_data, to_remove)
        print('Done')

    print('Training  Model..........')
    train_features = hr.concat(all_training_data, 'features')
    train_labels = hr.concat(all_training_data, 'labels')
    clf = train_model_forest(train_features, train_labels)
    print('Training  Model..........Done')

    print('Extracting Test Features..........')
    all_test_data = normalize_all(n_func, test)
    print('Extracting Test Features..........Done')

    if e_mode == 'melody':
        print('Removing Unvoiced Frames From Test..........', end='')
        all_test_data = only_voiced_frames(all_test_data)
        print('Done')

    print('Making Predictions..........', end='')
    predictions = predict(clf, all_test_data)
    print('Done')

    print('Exporting Predictions..........', end='')
    exporter.predictions(all_test_data, e_mode)
    print('Done')

    print('Evaluating Results..........')
    test_guesses = hr.concat(predictions, 'guesses')
    test_labels = hr.concat(predictions, 'labels')
    e_func(test_guesses, test_labels)
    print('Evaluating Results..........Done')


if __name__ == '__main__':
    main()
