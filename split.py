# File used for generating or loading train/test splits
# In practice, only final function should be used based on how the end user
# has seleted to split up the data

# ----------------- Imports
import medleydb as mdb
import os.path
import json
import exporter

# ----------------- Global Variables
train_test_split = 0.2  # Used for original dataset creation
train_validate_split = 0.2  # Change if you would like new validation split
train_test_name = 'train_test.json'


# ----------------- Functions
def quick_mode():
    '''
        Uses three short audio files for train and test data
        Should be used for testing that the system runs all the way through
        whithout crashing and not for estimating the strength of the features
        Outputs:
            List of medleydb tracks holding training data
            List of medleydb tracks holding test data
    '''
    train = [mdb.MultiTrack('MusicDelta_Reggae'),
             mdb.MultiTrack('MusicDelta_Rockabilly')]
    test = [mdb.MultiTrack('MusicDelta_Shadows')]
    return train, test


def validation_mode():
    '''
        Creates a random split of the training data into train/test
        Inputs:
            Float ranging from 0 to 1 referring to how to split data up
        Outputs:
            train: List of multitrack objects to be used for training
            test: List of multitrack objects to be used for validation
    '''
    # If we do not have a train/test set yet - create it!
    if not os.path.isfile(train_test_name):
        make_test_data()

    with open(train_test_name, 'r') as file:
        tt_data = json.load(file)

    melody_ids = tt_data['train']  # Do nothing with test data
    splits = mdb.utils.artist_conditional_split(trackid_list=melody_ids,
                                                test_size=train_validate_split,
                                                num_splits=1)

    train = [mdb.MultiTrack(t_id) for t_id in splits[0]['train']]
    test = [mdb.MultiTrack(t_id) for t_id in splits[0]['test']]
    return train, test


def test_mode():
    '''
        Loads the train_test data split from memory
        Outputs:
            train: List of multitrack objects to be used for training
            test: List of multitrack objects to be used for testing
    '''
    if not os.path.isfile(train_test_name):
        make_test_data()
        print('Uh oh, you are starting to test before any validation?!')
        print('Run again if you are really sure, but consider switching modes')
        quit()

    with open(train_test_name, 'r') as file:
        tt_data = json.load(file)
        train_ids = tt_data['train']  # Do nothing with test data
        test_ids = tt_data['test']

    train = [mdb.MultiTrack(t_id) for t_id in train_ids]
    test = [mdb.MultiTrack(t_id) for t_id in test_ids]
    return train, test


def make_test_data():
    '''
        Creates one pair of data and exports: train/test
        Should only be used once to generate train/test datasets at
        the beginning of research
        Outputs: JSON file called 'train_test.json'
    '''
    generator = mdb.load_melody_multitracks()
    melody_ids = [mtrack.track_id for mtrack in generator]
    splits = mdb.utils.artist_conditional_split(trackid_list=melody_ids,
                                                test_size=train_test_split,
                                                num_splits=1)

    train, test = splits[0]['train'], splits[0]['test']

    train_test_data = {'train': train, 'test': test}
    exporter.train_test(train_test_data, train_test_name)

    print('Generated Train/Test Data Split!')


# ----------------- Function Generator
def generate_split(split_type):
    '''
        Returns the right split function based on the string inputted
        If the string 'options' is inputted, it will return a dict containing
        modes rather than a function
        Input: String containing ['options' | 'voicing' | 'melody' | 'all']
        Output: Evaluation function corresponding to input
    '''
    splits_dict = {
        'quick': quick_mode,
        'validate': validation_mode,
        'test': test_mode
    }

    if split_type == 'options':
        return {i: k for i, k in enumerate(splits_dict)}
    else:
        return splits_dict[split_type]
