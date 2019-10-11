import pickle
import numpy as np
import os

class Agent():
    
    def __init__(
        self, 
        agent_name, 
        alpha=.6, 
        gamma=1, 
        reward_values={
            'win':1,
            'lose':-1,
            'draw':0.5
        }
    ):
        self.name = agent_name
        self.alpha = alpha # Learning rate
        self.gamma = gamma # discount factor
        self.rewards = reward_values
        self.load_qtable()

    def _init_qtable(self):
        init_config = [' ']*9
        hash_key = self.get_hash_index(init_config)
        init_q = {hash_key: [.5]*9}
        return init_q

    def _save_qtable(self):
        with open(os.getcwd()+'/agents/trained_qtables/'+self.name+'.pickle', 'wb') as handle:
            pickle.dump(self.qtable, handle)

    def load_qtable(self):
        try:
            with open(os.getcwd()+'/agents/trained_qtables/'+self.name+'.pickle', 'rb') as handle:
                self.qtable = pickle.load(handle)
        except:
            # Agent doesn't have an associoated qtable, initialize it and save it.
            self.qtable = self._init_qtable()
            self._save_qtable()

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
        weighted_prior_values = (1 - self.alpha) * current_q_value
        weighted_new_value = self.alpha * (reward + self.gamma * max_next_q_value)
        return weighted_prior_values + weighted_new_value

    def get_current_q_value(self, board_state, action):
        
        board_state_list = list(board_state.values())
        cur_config_hash_ix = self.get_hash_index(board_state_list)

        try:
            actions = self.qtable[cur_config_hash_ix]
            # Note: action is mapped from 1-9, and "actions" is an array whose indices are 0-8
            q_val = actions[action-1]
        except KeyError:
            q_val = 0.5
    
        return q_val

    def get_max_q_value(self, board_state):

        board_state_list = list(board_state.values())
        cur_config_hash_ix = self.get_hash_index(board_state_list)

        try:
            actions = self.qtable[cur_config_hash_ix]
            max_q_val = max(actions)
        except KeyError:
            max_q_val = 0.5
        
        return max_q_val

    def update_qvalue(self, board_state, action, new_qvalue):

        board_state_list = list(board_state.values())
        cur_config_hash_ix = self.get_hash_index(board_state_list)

        try:
            self.qtable[cur_config_hash_ix]
        except KeyError:
            self.qtable[cur_config_hash_ix] = [.5]*9

        # Note: action is mapped from 1-9, and "actions" is an array whose indices are 0-8
        self.qtable[cur_config_hash_ix][action-1] = new_qvalue

    def update_qvalues(self, machine_decision_states, outcome):
        
        reward = self.rewards[outcome]
        print("*************************")
        print("Updating QValues:")
        next_state = 0
        for state in reversed(machine_decision_states):

            print(state)
            print("\tNext State: "+str(next_state))
            cur_q_val = self.get_current_q_value(
                board_state = state[0],
                action = state[1]
            )
            print("\tCurrent Q value: "+str(cur_q_val))

            max_next_q_val = 0 if next_state == 0 else self.get_max_q_value(next_state) # 0 if it is the last decision

            new_q_val = self.calculate_new_q_value(
                current_q_value = cur_q_val,
                reward = reward,
                max_next_q_value = max_next_q_val
            )
            print("\tNew Q value: "+str(new_q_val))

            self.update_qvalue(
                board_state = state[0],
                action = state[1],
                new_qvalue = new_q_val
            )
            next_state = state[0]
        
        self._save_qtable()
        print(self.qtable)








