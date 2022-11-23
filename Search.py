import torch

import numpy as np

import Board
import Evaluator




def MCTS(state, evaluator, iters):

    root = Node(state)

    for i in range(iters):
        do_search(root, evaluator)

    return root


def do_search(root, evaluator):

    node = root
    while True:
        temp = node.explore()
        if temp == None: break
        node = temp

    val = node.evaluate_state(evaluator)

    node.back(val)


    return None


def choose_move(root):

    children = root.children
    vals = [child.get_exploitation() for child in children]

    best_child = children[np.argmax(vals)]

    return best_child.action, best_child.state, vals



# state must have:
# get_is_terminal(), evaluate(evaluator), get_states()

class Node():

    def __init__(self, state: Board.Board, parent=None, action=None):

        # set node state and parent node and move it took to get there
        self.state = state
        self.parent = parent
        self.action = action

        # init children list and if terminal
        self.children = None
        self.terminal_state = state.get_is_terminal()
        
        # init total scoring and number of visits
        self.cum_score = 0
        self.times = 0

    
    def get_val(self):
        val = np.inf
        if self.times > 0:
            val = self.get_exploitation() + self.get_exploration()
        return val

    def get_exploitation(self):
        return self.cum_score / self.times

    def get_exploration(self):
        c = np.sqrt(2)
        return c * np.sqrt(np.log(self.parent.times) / self.times)


    # returns None, unless it chooses a child node to explore
    def explore(self):

        # if terminal state, evaluate node
        if self.terminal_state:
            return None

        # if first time, add children and evaluate node
        elif self.times == 0:
            children_states, children_actions = self.state.get_states()
            self.children = [Node(children_states[i], self, children_actions[i]) for i in range(len(children_states))]
            return None

        # if been through before, add children and choose child node with highest val
        else:
            child = self.choose_child()
            return child


    def choose_child(self):
        vals = [child.get_val() for child in self.children]
        return self.children[np.argmax(vals)]

    def evaluate_state(self, evaluator):
        return self.state.get_terminal_value() if self.terminal_state else self.state.evaluate(evaluator).item()

    def back(self, val):

        self.times += 1
        self.cum_score += val

        if self.parent != None: self.parent.back(1-val)

        return None


        

