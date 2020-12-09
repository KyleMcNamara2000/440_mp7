import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.s = None
        self.a = None
        self.points = 0

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #state = [snake_head_x, snake_head_y, snake_body, food_x, food_y], 14x14
    def getStateTuple(self, state):
        #(adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        adjoining_wall_y = int(state[1] < 80) + 2 * int(state[1] >= 480)
        adjoining_wall_x = int(state[0] < 80) + 2 * int(state[0] >= 480)

        food_dir_x = int((state[0] - state[3]) < 0) * 2 + int((state[0] - state[3]) > 0)
        food_dir_y = int((state[1] - state[4]) < 0) * 2 + int((state[1] - state[4]) > 0)

        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        if (state[0], state[1] - 40) in state[2]:
            adjoining_body_top = 1
        if (state[0], state[1] + 40) in state[2]:
            adjoining_body_bottom = 1
        if (state[0] - 40, state[1]) in state[2]:
            adjoining_body_left = 1
        if (state[0] + 40, state[1]) in state[2]:
            adjoining_body_right = 1

        tup = (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return tup


    def R(self, points, dead):
        if dead:
            return -1
        if self.points != points:
            self.points = points
            return 1
        return -0.1

    def f(self, qVal, nVal):
        #f(u,n) returns 1 if n is less than a tuning parameter Ne, otherwise it returns u.
        #print(qVal, nVal)
        if nVal < self.Ne:
            return 1
        return qVal

    #grid: 560x560, 40 size squares, outside = walls
    #s' = state, s = self.s
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        sPrime = self.getStateTuple(state)
        action = self.a
        reward = self.R(points, dead)

        #Q(s,a)+α(R(s)+γmaxa′Q(s′,a′)−Q(s,a))
        if self._train is True:
            if action != None and self.s != None:
                #pick best action to take from this state s'
                maxNext = np.max(self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7]])
                maxNext *= self.gamma
                alpha = self.C / (self.C + self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], action])
                curr = self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], action]
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], action] += alpha * (reward + maxNext - curr)
            #pick best action from s' using f(x)
            bestAction = 0
            maxAction = -1000000
            for act in [3, 2, 1, 0]:
                curr = self.f(self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7], act], self.N[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7], act])
                #print(curr)
                if curr > maxAction:
                    maxAction = curr
                    bestAction = act
            #done
            #print(bestAction)
            if not dead:
                self.N[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7], bestAction] += 1
            self.s = sPrime
            self.a = bestAction
        else:
            #pick max action for Q
            bestAction = np.argmax(self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7]])
        #print(bestAction)
        if dead:
            self.reset()
        return bestAction


