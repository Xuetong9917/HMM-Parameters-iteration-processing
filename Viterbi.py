# COMP4121 main project, HMM-VA
# @Author: Xuetong Wang
# @Date: 21/11/2023

# Viterbi Algorithm for HMM
# dynamic programming, time complexity O(mn^2), m is the length of sequence of observation, n is the number of hidden states

states = ('1', '2', '3')

observations = ('A', 'B')

start_probability = {'1': 1, '2': 0, '3': 0}

transition_probability = {
    '1': {'1': 0.34, '2': 0.54, '3': 0.12},
    '2': {'1': 1, '2': 0, '3': 0},
    '3': {'1': 1, '2': 0, '3': 0}
}

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