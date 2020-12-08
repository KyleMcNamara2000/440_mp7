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
        adjoining_wall_y = int(state[1] <= 80) + 2 * int(state[1] >= 480)
        adjoining_wall_x = int(state[0] <= 80) + 2 * int(state[0] >= 480)
        food_dir_x = int((state[0] - state[3]) / 40 == -1) + 2 * int((state[0] - state[3]) / 40 == 1)
        food_dir_y = int((state[1] - state[4]) / 40 == -1) + 2 * int((state[1] - state[4]) / 40 == 1)
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for pt in state[2]:
            if pt == (state[0], state[1] - 40):
                adjoining_body_top = 1
            if pt == (state[0], state[1] + 40):
                adjoining_body_bottom = 1
            if pt == (state[0] - 40, state[1]):
                adjoining_body_left = 1
            if pt == (state[0] + 40, state[1]):
                adjoining_body_right = 1
        tup = (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return tup

    #TODO: why is points needed???
    def R(self, points, dead):
        if dead:
            return -1
        if self.points != points:
            self.points = points
            return 1
        return -0.1

    def f(self, qVal, nVal):
        #f(u,n) returns 1 if n is less than a tuning parameter Ne, otherwise it returns u.
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
        #TODO: update N/Q, what order???
        sPrime = self.getStateTuple(state)
        s = self.s
        action = self.a


        actions = [0, 1, 2, 3]
        #print("hi", self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7]])
        bestAction = np.argmax(self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7]])
        #print(bestAction)
        #Q(s,a)+α(R(s)+γmaxa′Q(s′,a′)−Q(s,a))


        #print(self.N)
        if self._train is True:
            if action != None and s != None:
                #print("hey")
                #print("hoi", self.N[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]])
                self.N[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], action] += 1
                reward = self.R(points, dead)
                #pick best action to take from this state s'
                maxNext = np.max(self.Q[sPrime[0], sPrime[1], sPrime[2], sPrime[3], sPrime[4], sPrime[5], sPrime[6], sPrime[7]])
                maxNext *= self.gamma
                alpha = self.C / (self.C + self.N[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], action])
                curr = self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], action]
                self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], action] += alpha * (reward + maxNext - curr)
                #self.N[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], action] += 1

        self.s = sPrime
        self.a = bestAction
        if dead:
            self.reset()
            return bestAction

        return bestAction


        '''
        #calculate best action + max(Q(s',a')
        bestAction = 0
        bestMax = 0
        for a in actions:
            curr = self.f(self.Q[sPrime][a], self.N[sPrime][a])
            if curr > bestMax:
                bestAction = a
                bestMax = curr

        R = self.R(points, dead)
        alpha = self.C / (self.C + self.N[s][bestAction])

        if not self._train:
            return bestAction
        self.Q[s][bestAction] = self.Q[s][bestAction] + alpha * (R + self.gamma * bestMax - self.Q[s][bestAction])
        if dead:
            self.reset()
            return bestAction
        self.N[s][bestAction] += 1
        return bestAction
        '''
