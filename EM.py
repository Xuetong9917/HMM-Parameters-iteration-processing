# COMP4121 main project, HMM parameter iteration processing
#  author: Xuetong Wang, time: 21/11/2023

# calculating alpha_t(i)
# return alpha dict
import numpy as np


def alpha_cal(alpha_dict, str, t, i):
    if t == 1:
        return starting_prob[i] * observation_prob[(i, str)]
    else:
        temp_sum = 0
        for j in observation_state:
            temp_sum += alpha_dict[(t - 1, j)] * transition_prob[(j, i)] * observation_prob[(i, str)]
        return temp_sum


def beta_cal(beta_dict, str, t, i):
    temp_sum = 0
    for j in observation_state:
        temp_sum += beta_dict[(t + 1, j)] * transition_prob[i, j] * observation_prob[(j, str)]
    return temp_sum


def calculation_func(seq_str):
    alpha_dict = {}
    beta_dict = {}
    delta_dict = {}
    s_dict = {}

    for t in range(1, len(seq_str) + 1):
        for i in observation_state:
            alpha_dict[(t, i)] = alpha_cal(alpha_dict, seq_str[t - 1], t, i)

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
                                         observation_prob[(j, seq_str[t])] * beta_dict[(t + 1, j)]) / P_O

    for t in range(1, len(seq_str) + 1):
        P_t_O = 0
        for i in observation_state:
            P_t_O += alpha_dict[(t, i)]
        for i in observation_state:
            s_dict[(t, i)] = alpha_dict[(t, i)] / P_t_O

    return delta_dict, s_dict, P_O


def iter_func():
    # the state sequence is AABABAB
    ABBA_sequence_str = 'AABABAB'
    ABBA_delta_dict, ABBA_s_dict, ABBA_P_O = calculation_func(ABBA_sequence_str)

    ABBA_c = 1

    # the likelihood
    L_h = (ABBA_c * np.log(ABBA_P_O))

    # for new starting probability
    I = ABBA_s_dict[(1, '1')] * ABBA_c
    J = ABBA_s_dict[(1, '2')] * ABBA_c
    W = ABBA_s_dict[(1, '3')] * ABBA_c
    sum_I_J = I + J + W
    new_starting_prob = {'1': I/(I+J+W), '2': J/(I+J+W), '3': W/(I+J+W)}

    # for new s->s, s->t, t->s, t->t
    K, L, O, M, N, U, Z, X, Y = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for t in range(1, len(ABBA_sequence_str)):
        K += ABBA_delta_dict[(t, '1', '1')] * ABBA_c
        L += ABBA_delta_dict[(t, '1', '2')] * ABBA_c
        O += ABBA_delta_dict[(t, '1', '3')] * ABBA_c
        M += ABBA_delta_dict[(t, '2', '1')] * ABBA_c
        N += ABBA_delta_dict[(t, '2', '2')] * ABBA_c
        U += ABBA_delta_dict[(t, '2', '3')] * ABBA_c
        Z += ABBA_delta_dict[(t, '3', '1')] * ABBA_c
        X += ABBA_delta_dict[(t, '3', '2')] * ABBA_c
        Y += ABBA_delta_dict[(t, '3', '3')] * ABBA_c

    new_transition_prob = {
        ('1', '1'): K / (K + L + O),
        ('1', '2'): L / (K + L + O),
        ('1', '3'): O / (K + L + O),
        ('2', '1'): M / (M + N + U),
        ('2', '2'): N / (M + N + U),
        ('2', '3'): U / (M + N + U),
        ('3', '1'): Z / (Z + X + Y),
        ('3', '2'): X / (Z + X + Y),
        ('3', '3'): Y / (Z + X + Y),
    }

    # for new output A, B in state s, t
    K = (ABBA_s_dict[(1, '1')] + ABBA_s_dict[(4, '1')]) * ABBA_c
    L = (ABBA_s_dict[(2, '1')] + ABBA_s_dict[(3, '1')]) * ABBA_c
    O = (ABBA_s_dict[(1, '2')] + ABBA_s_dict[(4, '2')]) * ABBA_c
    M = (ABBA_s_dict[(2, '2')] + ABBA_s_dict[(3, '2')]) * ABBA_c
    N = (ABBA_s_dict[(1, '3')] + ABBA_s_dict[(4, '3')]) * ABBA_c
    U = (ABBA_s_dict[(2, '3')] + ABBA_s_dict[(3, '3')]) * ABBA_c
    new_observation_prob = {
        ('1', 'A'): K / (K + L),
        ('1', 'B'): L / (K + L),
        ('2', 'A'): O / (M + O),
        ('2', 'B'): M / (M + O),
        ('3', 'A'): N / (U + N),
        ('3', 'B'): U / (U + N),
    }

    return L_h, new_starting_prob, new_transition_prob, new_observation_prob


# starting probability of s is 0.85, t is 0.15
starting_prob = {'1': 1/3, '2': 1/3, '3': 1/3}
observation_state = ['1', '2', '3']

observation_prob = {
    ('1', 'A'): 0.75,
    ('1', 'B'): 0.25,
    ('2', 'A'): 0.25,
    ('2', 'B'): 0.75,
    ('3', 'A'): 1/3,
    ('3', 'B'): 2/3
}

transition_prob = {
    ('1', '1'): 0.5,
    ('1', '2'): 0.25,
    ('1', '3'): 0.25,
    ('2', '1'): 0.25,
    ('2', '2'): 0.25,
    ('2', '3'): 0.5,
    ('3', '1'): 1/3,
    ('3', '2'): 1/3,
    ('3', '3'): 1/3
}
L_h = 0
iter_count = 0
while True:
    new_L_h, new_starting_prob, new_transition_prob, new_observation_prob = iter_func()
    if abs(L_h - new_L_h) < 0.001:
        print(iter_count)
        break
    L_h = new_L_h
    starting_prob = new_starting_prob
    transition_prob = new_transition_prob
    observation_prob = new_observation_prob
    iter_count += 1

for item in starting_prob.items():
    print(item)

for item in transition_prob.items():
    print(item)

for item in observation_prob.items():
    print(item)


