# Any exports done to memory contained in this program

# ----------------- Imports
import json
import helpers as hr


# ----------------- Types of Data to Export
def predictions(all_test_data, mode):  # Add Unique Title Info!
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
        if mode != 'voicing':  # Convert Note Names to Frequencies
            new_d['labels'] = (hr.note_to_hz_zeros(d['labels'])).tolist()
            new_d['guesses'] = hr.note_to_hz_zeros(d['guesses']).tolist()
        else:
            new_d['labels'] = d['labels'].tolist()
            new_d['guesses'] = d['guesses'].tolist()

        new_d['t_id'] = d['t_id']
        new_d['times'] = d['times']
        copy.append(new_d)

    with open('predict.json', 'w') as file:
        json.dump(copy, file)
