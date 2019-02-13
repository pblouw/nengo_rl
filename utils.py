import numpy as np


def weight_init(shape):
    '''Convenience function for randomly initializing weights'''
    weights = np.random.uniform(-0.000001, 0.000001, size=shape)
    return weights

def sensor(t):
    '''Return current x,y coordinates of agent'''
    data = np.zeros(25)
    idx = 5 * (agent.x-1) + (agent.y -1)
    data[idx] = 1

    return data

def reward(t):
    return agent.reward

def compute_position(action_idx):
    if action_idx == 0:
        # move up
        x_pos = agent.x
        y_pos = agent.y - 1
    elif action_idx == 1:
        # move right
        x_pos = agent.x + 1
        y_pos = agent.y
    elif action_idx == 2:
        # move down
        x_pos = agent.x
        y_pos = agent.y + 1
    else:
        # move left
        x_pos = agent.x - 1
        y_pos = agent.y
    
    return x_pos, y_pos


def take_action(index, epsilon=0.1):
    '''Pick an action to perform in the environment'''
    do_random = np.random.choice([1, 0], p=[epsilon, 1-epsilon])
    if do_random:
        action_idx = np.random.choice(np.arange(4))    
    else:
        action_idx = index
    
    x_pos, y_pos = compute_position(action_idx)
    agent.set_position(x_pos, y_pos)
    return action_idx

class Values(object):
    def __init__(self, dimensions=8):
        self.output = np.zeros(dimensions)
        self.last_action_index = 0
        self.current_action_index = 0
        self.terminal = False
        self.terminal_clock = stepsize

    def step(self, t, x):
        '''Prepare Q value info for computing error signal'''
        # if t is multiple of action step size, do step 
        if self.terminal:
            self.terminal_clock -= 1
            if self.terminal_clock == 0:
                self.terminal = False
                self.terminal_clock = stepsize
            return np.zeros_like(self.output)
        
        if int(t * 1000) == 1:
            print('STARTING')

        if int(t * 1000) % stepsize == 0:
            if agent.reward > 0:
                self.output = np.zeros_like(self.output)
                x_pos = np.random.choice([1,5])
                y_pos = np.random.choice([1,5])
                agent.set_position(x_pos, y_pos)
                
                self.terminal = True
                return self.output
                
            self.last_action_index = self.current_action_index
            
            qs = self.output[8:]
            idx = np.argmax(qs)
            self.current_action_index = take_action(idx)

        # then on next step store new qvalues
        elif int(t * 1000) % stepsize == 1:
            qvalues = x
            qmax = qvalues[np.argmax(qvalues)]
            # set output to be current state q, qmax, last state selection
            
            c_output = np.zeros(4)
            c_output[self.current_action_index] = qvalues[self.current_action_index]
            
            f_output = np.zeros(4)
            f_output[self.current_action_index] = 0.9 * qmax + agent.reward
            
            self.output = np.concatenate(
                (c_output, f_output, qvalues))
            
        return self.output
