# COMP4121 main project, HMM parameter iteration processing
#  author: Xuetong Wang, time: 21/11/2023

# calculating alpha_t(i)
# return alpha dict
import numpy as np

def alpha_cal(alpha_dict, str, t, i):
    if t == 1:
        return starting_prob[i]*observation_prob[(i, str)]
    else:
        temp_sum = 0
        for j in observation_state:
            temp_sum += alpha_dict[(t-1, j)] * transition_prob[(j, i)] * observation_prob[(i, str)]
        return temp_sum


def beta_cal(beta_dict, str, t, i):
    temp_sum = 0
    for j in observation_state:
        temp_sum += beta_dict[(t+1, j)] * transition_prob[i, j] * observation_prob[(j, str)]
    return temp_sum



def calculation_func(seq_str):
    alpha_dict = {}
    beta_dict = {}
    delta_dict = {}
    s_dict = {}

    for t in range(1, len(seq_str)+1):
        for i in observation_state:
            alpha_dict[(t, i)] = alpha_cal(alpha_dict, seq_str[t-1], t, i)

    for t in range(len(seq_str), 0, -1):
        for i in observation_state:
            if t == len(seq_str):
                beta_dict[(t, i)] = 1
            else:
                beta_dict[(t, i)] = beta_cal(beta_dict, seq_str[t], t, i)

    P_O = 0
    for i in observation_state:
        P_O += alpha_dict[(len(seq_str), i)]

    for t in range(1, len(seq_str)):
        for i in observation_state:
            for j in observation_state:
                delta_dict[(t, i, j)] = (alpha_dict[(t, i)] * transition_prob[(i, j)] *
                                         observation_prob[(j, seq_str[t])] * beta_dict[(t+1, j)])/P_O

    for t in range(1, len(seq_str)+1):
        P_t_O = 0
        for i in observation_state:
            P_t_O += alpha_dict[(t, i)]
        for i in observation_state:
            s_dict[(t, i)] = alpha_dict[(t, i)] / P_t_O

    return delta_dict, s_dict, P_O



def iter_func(starting_prob, observation_state, observation_prob, transition_prob):
    # the state sequence is ABBA
    ABBA_sequence_str = 'ABBA'
    ABBA_delta_dict, ABBA_s_dict, ABBA_P_O = calculation_func(ABBA_sequence_str)
    # the state sequence is BAB
    BAB_sequence_str = 'BAB'
    BAB_delta_dict, BAB_s_dict, BAB_P_O = calculation_func(BAB_sequence_str)

    ABBA_c = 10
    BAB_c = 20

    # the likelihood
    L_h = (ABBA_c * np.log(ABBA_P_O)) + (BAB_c * np.log(BAB_P_O))

    # for new starting probability

    I = ABBA_s_dict[(1, 's')]*ABBA_c + BAB_s_dict[(1, 's')]*BAB_c
    J = ABBA_s_dict[(1, 't')]*ABBA_c + BAB_s_dict[(1, 't')]*BAB_c
    sum_I_J = I+J
    I = I / sum_I_J
    J = J / sum_I_J
    new_starting_prob = {'s': I, 't': J}

    # for new s->s, s->t, t->s, t->t
    K, L, M, N = 0, 0, 0, 0
    for t in range(1, len(ABBA_sequence_str)):
        K += ABBA_delta_dict[(t, 's', 's')] * ABBA_c
        L += ABBA_delta_dict[(t, 's', 't')] * ABBA_c
        M += ABBA_delta_dict[(t, 't', 's')] * ABBA_c
        N += ABBA_delta_dict[(t, 't', 't')] * ABBA_c

    for t in range(1, len(BAB_sequence_str)):
        K += BAB_delta_dict[(t, 's', 's')] * BAB_c
        L += BAB_delta_dict[(t, 's', 't')] * BAB_c
        M += BAB_delta_dict[(t, 't', 's')] * BAB_c
        N += BAB_delta_dict[(t, 't', 't')] * BAB_c

    new_transition_prob = {
        ('s', 's'): K / (K + L),
        ('s', 't'): L / (K + L),
        ('t', 's'): M / (M + N),
        ('t', 't'): N / (M + N)
    }

    # for new output A, B in state s, t
    K, L, M, N = 0, 0, 0, 0
    K = (ABBA_s_dict[(1, 's')] + ABBA_s_dict[(4, 's')]) * ABBA_c + BAB_s_dict[(2, 's')] * BAB_c
    L = (ABBA_s_dict[(2, 's')] + ABBA_s_dict[(3, 's')]) * ABBA_c + (BAB_s_dict[(1, 's')] + BAB_s_dict[(3, 's')]) * BAB_c
    M = (ABBA_s_dict[(1, 't')] + ABBA_s_dict[(4, 't')]) * ABBA_c + BAB_s_dict[(2, 't')] * BAB_c
    N = (ABBA_s_dict[(2, 't')] + ABBA_s_dict[(3, 't')]) * ABBA_c + (BAB_s_dict[(1, 't')] + BAB_s_dict[(3, 't')]) * BAB_c
    new_observation_prob = {
        ('s', 'A'): K / (K + L),
        ('s', 'B'): L / (K + L),
        ('t', 'A'): M / (M + N),
        ('t', 'B'): N / (M + N)
    }


    return L_h, new_starting_prob, new_transition_prob, new_observation_prob


# starting probability of s is 0.85, t is 0.15
starting_prob = {'s': 0.85, 't': 0.15}
observation_state = ['s', 't']

observation_prob = {
    ('s', 'A'): 0.4,
    ('s', 'B'): 0.6,
    ('t', 'A'): 0.5,
    ('t', 'B'): 0.5
}

transition_prob = {
    ('s','s'): 0.3,
    ('s', 't'): 0.7,
    ('t', 's'): 0.1,
    ('t', 't'): 0.9
}
L_h = 0
iter_count = 0
while True:
    new_L_h, new_starting_prob, new_transition_prob, new_observation_prob = \
        iter_func(starting_prob, observation_state, observation_prob, transition_prob)
    if abs(L_h - new_L_h) < 0.001:
        print(iter_count)
        break
    L_h = new_L_h
    starting_prob = new_starting_prob
    transition_prob = new_transition_prob
    observation_prob = new_observation_prob
    iter_count += 1
