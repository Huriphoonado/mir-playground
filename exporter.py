# Any exports done to memory contained in this file

# ----------------- Imports
import json
import helpers as hr
import os


# ----------------- Types of Data to Export
def predictions(all_test_data, mode, file_name):
    '''
        Exports the predictions dict into a json file so that results
        be loaded and graphed in another program
        Inputs:
            List of dicts containing all test results
            String containing ['voicing' | 'melody' | 'all']
        Outputs:
            JSON File
    '''
    copy = []
    for d in all_test_data:
        new_d = {}
        if mode != 'voicing':  # Convert Note Names to MIDI vals for plotting
            new_d['labels'] = (hr.note_to_midi_zeros(d['labels'])).tolist()
            new_d['guesses'] = hr.note_to_midi_zeros(d['guesses']).tolist()
        else:
            new_d['labels'] = d['labels'].tolist()
            new_d['guesses'] = d['guesses'].tolist()

        new_d['t_id'] = d['t_id']
        new_d['times'] = d['times']
        copy.append(new_d)

    # Predictions written to a results directory, so make it if one does not
    # yet exist
    if not os.path.exists('results'):
        os.makedirs('results')

    with open(file_name, 'w') as file:
        json.dump(copy, file)


def train_test(train_test_data, train_test_name):
    '''
        Simply writes train/test split to a dictionary stored in melody
        Inputs:
            Dict containing the train/test split
            Name to write the file as (Should be based on the task)
        Output:
            JSON File
    '''
    with open(train_test_name, 'w') as file:
        json.dump(train_test_data, file)
