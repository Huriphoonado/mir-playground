# File containing functions used for evaluation of all data
# In practice, only the run_eval function should be used given a string
# variable in the main function pertaining to the mode

# Functions convert to cents and voicing following mirex guidlines:
#    http://craffel.github.io/mir_eval/#module-mir_eval.melody

# ----------------- Imports
from mir_eval import melody as mel_eval
import numpy as np
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
    correct_tries = (ref_voicing == est_voicing)
    overall_accuracy = sum(correct_tries)/correct_tries.size

    metrics = {
        'vx_recall': vx_recall,
        'vx_false_alarm ': vx_false_alarm,
        'overall_accuracy': overall_accuracy
    }

    for m, v in metrics.items():  # Python2 is iteritems I think
        print(m, ':', v)
    return metrics


def evaluate_model_melody(test_guesses, test_labels):
    ref_freq = hr.note_to_hz_zeros(test_labels)
    est_freq = hr.note_to_hz_zeros(test_guesses)

    ref_cent = mel_eval.hz2cents(ref_freq)
    est_cent = mel_eval.hz2cents(est_freq)

    all_voiced = np.ones(len(ref_cent), dtype=bool)

    print('Evaluating pitch...')
    raw_pitch = mel_eval.raw_pitch_accuracy(all_voiced, ref_cent,
                                            all_voiced, est_cent,
                                            cent_tolerance=50)

    print('Evaluating chroma...')
    raw_chroma = mel_eval.raw_chroma_accuracy(all_voiced, ref_cent,
                                              all_voiced, est_cent,
                                              cent_tolerance=50)
    metrics = {
        'raw_pitch': raw_pitch,
        'raw_chroma': raw_chroma,
    }

    for m, v in metrics.items():
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
        Input: String containing ['options' | 'voicing' | 'melody' | 'all']
        Output: Evaluation function corresponding to input
    '''
    evaluations = {
        'voicing': evaluate_model_voicing,
        'melody': evaluate_model_melody,
        'all': evaluate_model_all
    }

    if mode == 'options':
        return {i: k for i, k in enumerate(evaluations)}
    else:
        return evaluations[mode]
