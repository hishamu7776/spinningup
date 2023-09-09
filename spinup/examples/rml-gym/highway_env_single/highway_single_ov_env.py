import gym
import cv2 
import time

import numpy as np 
import PIL.Image as Image
import matplotlib.pyplot as plt
from gym import Env, spaces

def make() -> "HighwaySingleEnvironment":
    return HighwaySingleEnvironment()


class HighwaySingleEnvironment(Env):
    def __init__(self):
        super(HighwaySingleEnvironment, self).__init__()
        

        # Define a 2-D observation space
        self.observation_shape = (1,3)
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0]), 
                                            high = np.array([1,1,1,1,1]),
                                            dtype = np.float16)
    
        # STATE LEGEND: [risk mine, risk left, risk right, lane]
        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(3,)
        
        self.max_speed = 20
        self.min_speed = 15
        self.car_count = 0
        self.elements = []
        self.observations = []
        self.actions = []
        self.t = 0
        self.f = open("history.txt", "w")
        self.f.close()
        
    def getObs(self):
        self.most_dangerous = None
        lane = self.ego.getState()[0]
        obs = [500, 500, 500, lane-2, (self.ego.getState()[2]-self.min_speed)/(self.max_speed-self.min_speed)]
        
        if lane == 2:
            obs[2] = 0
        elif lane == 3:
            obs[1] = 0
            
        for elem in self.elements:
            if isinstance(elem, Car):
                interdist = elem.getState()[1] - self.ego.getState()[1]
                diff_vel = elem.getState()[2] - self.ego.getState()[2]
                if interdist*diff_vel <= 0:
                    coll_time = (abs(interdist)+1)/(abs(diff_vel) + 1)
                else:
                    coll_time = abs(interdist)+1
                    
                if elem.getState()[0] == lane:
                    if obs[0] > coll_time:
                        obs[0] = coll_time
                        self.most_dangerous = elem
                elif elem.getState()[0] == lane + 1:
                    if obs[1] > coll_time:
                        obs[1] = coll_time
                elif elem.getState()[0] == lane - 1:
                    if obs[2] > coll_time:
                        obs[2] = coll_time
                        
        obs[0] = np.log(obs[0]+1)/np.log(501)
        obs[1] = np.log(obs[1]+1)/np.log(501)
        obs[2] = np.log(obs[2]+1)/np.log(501)
                    
        return obs
                
        
    def reset(self):
        # Reset the reward
        self.ep_return  = 0
        self.t = 0
        # Initialise the ego vehicle
        self.ego = Ego()
        self.car1 = Car(2, 100, 15)
        self.car2 = Car(3, -20, 23)
        self.car3 = Car(3, -100, 23)
        self.car4 = Car(3, -200, 23)
        self.most_dangerous = None
        
        self.car_count = 4
        
        # Initialise the elements 
        self.elements = [self.ego, self.car1, self.car2, self.car3, self.car4]
        
        self.obs = self.getObs()
        return self.obs
        
    def get_action_meanings(self):
        return {0: "Left", 1: "Right", 2: "Maintain"}
    
    def has_collided(self, elem1, elem2):
        if elem1.getState()[0] == elem2.getState()[0]:
            if abs(elem1.getState()[1] - elem2.getState()[1]) < 2.5:
                return True
        return False
    
    def step(self, action):
        t = 1
        self.t = self.t + 1
        # Flag that marks the termination of an episode
        done = False
        
        reward = 0
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"
        
        if action == 0:
            self.a = "left"
        elif action == 1:
            self.a = "right"
        else:
            self.a = "maintain"
            
        self.actions.append(self.a)

        # apply the action to the chopper
        if action == 0:
            if self.ego.getState()[0] == 3:
                done = True
                # reward += self.ego.getDistance()
            else:
                self.ego.left()
        elif action == 1:
            if self.ego.getState()[0] == 2:
                done = True
                # reward += self.ego.getDistance()
            else:
                if self.obs[0] < self.obs[1]:
                    reward += 50
                self.ego.right()
                
        elif action == 2:
            if self.most_dangerous == None:
                self.ego.accelerate()
            else:
                reward += 2
                interdist = self.most_dangerous.getState()[1] - self.ego.getState()[1]
                diff_vel = self.most_dangerous.getState()[2] - self.ego.getState()[2]
                if interdist > 0 and diff_vel < 0:
                    if abs((interdist + 5)/diff_vel) < 2*self.ego.getState()[2]:
                        self.ego.brake()
                    else:
                        self.ego.accelerate()
                elif interdist > 0 and diff_vel > 0:
                    self.ego.accelerate()
                elif interdist < 0 and diff_vel > 0:
                    if abs((interdist - 5)/diff_vel) < 2*self.ego.getState()[2]:
                        self.ego.accelerate()
        
        
        self.obs = self.getObs()
        self.observations.append(self.obs)
        
        # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Car):
                elem.tick(t, self.ego.getState()[2])
                if elem.getState()[1] < -500 or elem.getState()[1] > 500:
                    self.elements.remove(elem)
                    self.car_count = self.car_count - 1
                
                # If the car has collided.
                if self.has_collided(self.ego, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    self.f = open("history.txt", "a")
                    self.f.write("Crashed\n")
                    self.f.write(str(elem.getState()))
                    self.f.write("\n")
                    self.f.write(str(self.ego.getState()))
                    self.f.write("\n")
                    if not(self.most_dangerous == None):
                        self.f.write(str(self.most_dangerous.getState()))
                        self.f.write("\n")
                    for i in range(1, min(10, self.t - 1)):
                        self.f.write(str(self.t - i))
                        self.f.write("\n")
                        self.f.write(str(self.actions[self.t - i]))
                        self.f.write("\n")
                        self.f.write(str(self.observations[self.t - i]))
                        self.f.write("\n")
                    self.f.write("++++++++++++++++++++++++++++++++++++++++\n")
                    self.f.close()
                    # reward += self.ego.getDistance()
                    # self.elements.remove(self.ego)
            else:
                elem.tick(t)

        # Increment the episodic return
        self.ep_return += 1
        
        if self.t > 100:
            done = True
            reward = self.ego.getDistance()/10
            self.f = open("history.txt", "a")
            self.f.write("Well done\n")
            self.f.write(str(self.ego.getDistance()))
            self.f.write("\n")
            self.f.write("---------------------------------------\n")
            # reward += self.ego.getDistance()
            self.f.close()
        
        reward += 5*self.obs[0]
        reward += 2*self.obs[4]
    
        return self.obs, reward, done, []
    
    def printEnv(self, action):
        fig, ax = plt.subplots()
        for elem in self.elements:
            if isinstance(elem, Ego):
                ax.plot(elem.getState()[1], elem.getState()[0], 'ro')
            else:
                ax.plot(elem.getState()[1], elem.getState()[0], 'bo')
        ax.set_ylim(0, 5)
        ax.set_xlim(-500, 500)
        plt.show()
        print(self.a)
        print(self.obs)
        print(self.ego.getState()[2])
        time.sleep(0.001)

class Car(object):
    def __init__(self, lane, position, velocity):
        self.lane = lane
        self.position = position
        self.velocity = velocity
        
    def getState(self):
        return np.array([self.lane, self.position, self.velocity])
        
    def tick(self, t, vel):
        self.position = self.position - (vel - self.velocity)*t
        
        
class Ego(object):
    def __init__(self):
        self.lane = 2
        self.position = 0
        self.velocity = 20
        self.distance = 0
        
    def getState(self):
        return np.array([self.lane, self.position, self.velocity])
        
    def left(self):
        if self.lane < 4:
            self.lane += 1
        
    def right(self):
        if self.lane > 1:
            self.lane -= 1
        
    def accelerate(self):
        if self.velocity < 20:
            self.velocity += 1
        
    def brake(self):
        if self.velocity > 15:
            self.velocity -= 1
            
    def getDistance(self):
        return self.distance
    
    def tick(self, t):
        self.distance = self.distance + self.velocity*t