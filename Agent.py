import pickle
import numpy as np
import os

class Agent():
    
    def __init__(self, agent_name, alpha=.6, gamma=1):
        self.name = agent_name
        self.qtable = self.load_qtable()
        self.alpha = alpha # Learning rate
        self.gamma = gamma # discount factor

    def _init_qtable(self):
        init_config = [' ']*9
        hash_key = self.get_hash_index(init_config)
        init_q = {hash_key: [.5]*9}
        with open(os.getcwd()+'/agents/trained_qtables/'+self.name+'.pickle', 'wb') as handle:
            pickle.dump(init_q, handle)

    def load_qtable(self):
        try:
            with open(os.getcwd()+'/agents/trained_qtables/'+self.name+'.pickle', 'rb') as handle:
                self.qtable = pickle.load(handle)
        except:
            self.qtable = self._init_qtable()

    def get_hash_index(self, board_configuration):
        """
        Hash function is defined as:

        ix = (3^0)*v0 + (3^1)*v1 (3^2)*v2 + ... +(3^8)*v8
        where Vi = 0 if ' ', 1 if 'X', 2 if 'O'

        Example of a board configuration:
        ['X', ' ', 'O', ' ', 'O', 'X', ' ', ' ', ' ']
        is equivalent to:
        
        X| |O
        -|-|-
         |O|X
        -|-|-
         | | 
        """
        values_map = {' ':0, 'X':1, 'O':2}
        exp = np.array([3**i for i in range(9)])
        vals = np.array([values_map[z] for z in board_configuration])
        ix = exp.dot(vals)
        return ix

    def calculate_new_q_value(
        self,
        current_q_value,
        reward, 
        max_next_q_value
    ):
        weighted_prior_values = (1 - learning_rate) * current_q_value
        weighted_new_value = self.alpha * (reward + self.gamma * max_next_q_value)
        return weighted_prior_values + weighted_new_value

    def get_current_q_value(self, board_state, action):
        
        board_state_list = list(board_state.values())
        cur_config_hash_ix = agent.get_hash_index(board_state_list)

        actions = self.qtable[cur_config_hash_ix]
    
        return actions[action]

    
    def get_max_q_value(self, board_state):

        board_state_list = list(board_state.values())
        cur_config_hash_ix = agent.get_hash_index(board_state_list)

        actions = self.qtable[cur_config_hash_ix]
        return max(actions)

    def update_qvalue(self, board_state, action, new_qvalue):

        board_state_list = list(board_state.values())
        cur_config_hash_ix = agent.get_hash_index(board_state_list)

        self.qtable[cur_config_hash_ix][action] = new_qvalue
 
    def update_qvalues(self, machine_decision_states, outcome):
        if outcome == 'lost':
            reward = -1
        elif outcome == 'win':
            reward = 1
        else:
            reward = 0.5

        next_state = 0
        for state in reversed(machine_decision_states):

            cur_q_val = self.get_current_q_value(
                board_state = machine_decision_states[0],
                action = machine_decision_states[1]
            )

            max_next_q_val = 0 if next_state == 0 else self.get_max_q_value(next_state) # 0 if it is the last decision

            new_q_val = self.calculate_new_q_value(
                current_q_value = cur_q_val,
                reward = reward,
                max_next_q_value = max_next_q_val
            )

            self.update_qvalue(
                board_state = machine_decision_states[0],
                action = machine_decision_states[1]
            )

            next_state = state







