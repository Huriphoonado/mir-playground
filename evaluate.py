# File containing functions used for evaluation of all data
# In practice, only the run_eval function should be used given a string
# variable in the main function pertaining to the mode

# Functions convert to cents and voicing following mirex guidlines:
#    http://craffel.github.io/mir_eval/#module-mir_eval.melody

# ----------------- Imports
from mir_eval import melody as mel_eval
import helpers as hr


# ----------------- Evaluation Functions
def evaluate_model_voicing(test_guesses, test_labels):
    '''
        If we are only looking at voicing, then we only care about some metrics
        Inputs:
            1d Boolean np.array containing all predictions made by the model
            1d Boolean np.array containing all ground truth labels
        Output: Dict containing
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
        Inputs:
            1d np.array containing all predictions made by the model
            1d np.array containing all ground truth labels
        Outputs:
            Dict holding results of all evaluations
    '''
    print('Running conversions...')
    ref_freq = hr.note_to_hz_zeros(test_labels)  # And back to Hz!
    est_freq = hr.note_to_hz_zeros(test_guesses)

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


def generate_eval(mode):
    '''
        Returns the right evaluation function based on the string inputted
        Input: String containing ['voicing' | 'melody' | 'all']
        Output: Evaluation function corresponding to input
    '''
    evaluations = {
        'all': evaluate_model_all,
        'voicing': evaluate_model_voicing
    }

    return evaluations[mode]
