
def get_SR_inds(RUN_ID):

    chosen_all_inds = {
        'SP': {
            'R0': 8,   # JN
            'R1': 8,   # JH
            'C1': 7,   # JN
            'k': 7,    # JH
            's': 6     # JN
        }
    }

    best_all_inds = {
        'SP': {
            'R0': 19,  # JN
            'R1': 19,  # JH
            'C1': 24,  # JN
            'k': 22,   # JH
            's': 27    # JN
        }
    }

    return chosen_all_inds[RUN_ID], best_all_inds[RUN_ID]


def get_ref_values():
    REF_VALUES = {
    'R0': 0.01,
    'R1': 0.01,
    'C1': 1000,
    'k':  0.01,
    's':  0.37266314,
    'sdot': 0.0001, # = 
    'C': 5,
    'd': 30,
    }
    return REF_VALUES