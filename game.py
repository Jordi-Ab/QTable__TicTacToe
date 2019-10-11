import numpy as np
from copy import deepcopy
import pickle
import os
from Agent import Agent
from collections import Counter

WINNING_COMBINATIONS=[
    [1,2,3],[4,5,6],[7,8,9], # horizontals
    [1,4,7],[2,5,8],[3,6,9], # verticals
    [1,5,9],[3,5,7] # diagonals
]

def win(board_state, value):
    for comb in WINNING_COMBINATIONS:
        val = 0
        for v in comb:
            val += 1 if board_state[v] == value else 0
        if val == 3:
            return True
    return False

def no_moves_available(board_state):
    cnt = Counter(list(board_state.values()))
    if cnt.get(' '):
        return False
    return True

def dict_to_matrix(board_state):
    m = []
    l = []
    for key, val in board_state.items(): 
        if key % 3 != 0:
            l.append(val)
        else:
            l.append(val)
            m.append(l)
            l=[]
    return m

def print_board(board_state):
    print('Current status of the game:')
    board_state_array = dict_to_matrix(board_state)
    for i, e in enumerate(board_state_array):
        print('|'.join(e))
        print('-|-|-' if (i+1)%3 !=0 else ' ')

def assert_correct_selection(selection, board_state):
    s = int(selection)
    try:
        if board_state[s] != ' ':
            raise ValueError
        return s
    except KeyError:
        raise ValueError

def human_turn(board_state):
    selection = input("Select a position, from 1 to 9, to place an 'X': ")
    try:
        sel = assert_correct_selection(selection, board_state)
        # Update board
        board_state[sel] = 'X'
        print_board(board_state)
    except ValueError:
        print("Invalid selection")
        human_turn(board_state)

def init_board():
    board_state = {1:' ',2:' ',3:' ',4:' ',5:' ',6:' ',7:' ',8:' ',9:' '}
    return board_state

def machine_turn(board_state, agent, machine_decision_states):
    
    board_state_list = list(board_state.values())
    cur_config_hash_ix = agent.get_hash_index(board_state_list)
    
    try:
        cur_config_qvals = agent.qtable[cur_config_hash_ix]
        print(cur_config_qvals)
    except KeyError:
        # Random move
        print('agent played randomly')
        cur_config_qvals = [.5]*9
    
    selection = make_move(board_state_list, cur_config_qvals)
    
    state_action_tuple = deepcopy(board_state), selection
    machine_decision_states.append(state_action_tuple)

    board_state[selection] = 'O'
    print_board(board_state)

def make_move(board_state_list, cur_config_qvals):
    # replace with nans on the places where a move has already been made
    mask_available = np.where(
        np.array(board_state_list)!=' ', 
        np.nan, 
        np.array(cur_config_qvals)
    )

    # Indices of max qvalues
    max_ixs = np.where(
        mask_available == np.nanmax(mask_available)
    )[0]

    # Random choice between the max q values
    move = np.random.choice(max_ixs)
    return move+1

def load_agents():
    agents = []
    for file in os.listdir('agents'):
        if file.endswith('pickle'):
            with open('agents/'+file, 'rb') as handle:
                A = pickle.load(handle)
                agents.append(A)
    return agents

def list_agents(agents_list):
    as_str = "0) New untrained agent"
    for i, A in enumerate(agents_list):        
        as_str += '\n'+str(i+1)+') '+A.name
    return as_str

def ask_for_agent_to_play_against():
    agents_list = load_agents()
    print(list_agents(agents_list))
    while(True):
        answ = input('Which agent would you like to play against? ')
        available_answers = [str(a) for a in range(len(agents_list)+1)]
        if answ not in available_answers:
            print("Not an available agent.")
        elif answ == '0':
            print("You chose to create a new Agent.")
            name = input("Give a name to this Agent: ")
            new_agent = Agent(name)
            with open('agents/'+new_agent.name+'.pickle', 'wb') as handle:
                pickle.dump(new_agent, handle)
            return new_agent
        else:
            Ag = agents_list[int(answ)-1]
            print("You chose to play against "+ Ag.name)
            print('Good Luck')
            Ag.load_qtable()
            print(Ag.qtable)
            return Ag

def main():
    agent = ask_for_agent_to_play_against()
    board_state = init_board()
    machine_decision_states = []
    print(
        """
Welcome to the Tic Tac Toe, to win complete a straight line
of your letter (Diagonal, Horizontal, Vertical). The board
has positions 1-9 starting at the top left.
        """
    )
    print_board(board_state)
    outcome = 'draw'
    while True:
        human_turn(board_state)
        if win(board_state, 'X'):
            print('Human wins')
            outcome = 'lose'
            break
        elif no_moves_available(board_state):
            print("It's a draw")
            outcome='draw'
            break

        machine_turn(board_state, agent, machine_decision_states)
        if win(board_state, 'O'):
            print('Agent wins')
            outcome = 'win'
            break
        elif no_moves_available(board_state):
            print("It's a draw")
            outcome='draw'
            break
    print(machine_decision_states)
    agent.update_qvalues(machine_decision_states, outcome)

if __name__ == '__main__':
    main()