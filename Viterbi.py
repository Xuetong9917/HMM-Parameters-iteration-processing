# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-04-02 08:52:24
# @Last Modified by:   WuLC
# @Last Modified time: 2017-04-02 09:50:31

###########################################################################################################
# Viterbi Algorithm for HMM
# dynamic programming, time complexity O(mn^2), m is the length of sequence of observation, n is the number of hidden states
# more details can be obtained from: http://wulc.me/2017/03/02/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95/
##########################################################################################################


# five elements for HMM
# states = ('Healthy', 'Fever')
states = ('1', '2', '3')

# observations = ('normal', 'cold', 'dizzy')
observations = ('A', 'B')

# start_probability = {'Healthy': 0.6, 'Fever': 0.4}
start_probability = {'1': 1, '2': 0, '3': 0}

# transition_probability = {
#     'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
#     'Fever': {'Healthy': 0.4, 'Fever': 0.6},
# }
transition_probability = {
    '1': {'1': 0.34, '2': 0.54, '3': 0.12},
    '2': {'1': 1, '2': 0, '3': 0},
    '3': {'1': 1, '2': 0, '3': 0}
}

# emission_probability = {
#     'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
#     'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
# }
emission_probability = {
    '1': {'A': 0.63, 'B': 0.37},
    '2': {'A': 0, 'B': 1},
    '3': {'A': 0, 'B': 1}
}


def Viterbit(obs, states, s_pro, t_pro, e_pro):
    path = {s: [] for s in states}  # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in states:
        curr_pro[s] = s_pro[s] * e_pro[s][obs[0]]
    for i in range(1, len(obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:
            max_pro, last_sta = max(
                ((last_pro[last_state] * t_pro[last_state][curr_state] * e_pro[curr_state][obs[i]], last_state)
                 for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    # find the final largest probability
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
    # print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path


if __name__ == '__main__':
    # obs = ['normal', 'cold', 'dizzy']
    obs = ['A', 'A', 'B', 'A', 'B', 'A', 'B']
    max_path = Viterbit(obs, states, start_probability, transition_probability, emission_probability)
    print(max_path)