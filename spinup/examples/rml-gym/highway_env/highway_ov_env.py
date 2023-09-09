import gym
import cv2 
import time
import random

import numpy as np 
import PIL.Image as Image
import matplotlib.pyplot as plt
from gym import Env, spaces

def make(env=None) -> "HighwayEnvironment":
    return HighwayEnvironment(env)

class HighwayEnvironment(Env):
    def __init__(self):
        super(HighwayEnvironment, self).__init__()
        

        # Define a 2-D observation space
        self.observation_shape = (1,6)
        self.observation_space = spaces.Box(low = np.array([0,0,0,0,0,0]), 
                                            high = np.array([1,1,1,1,1,1]),
                                            dtype = np.float16)
    
        # STATE LEGEND: [lane, velocity-target, safety front, safety back, safety right, safety left]
        # Define an action space ranging from 0 to 2
        self.action_space = spaces.Discrete(3,)
        self.max_speed = 25
        self.min_speed = 15
        self.car_count = 0
        self.elements = []
        self.observations = []
        self.actions = []
        self.t = 0
        self.f = open("history.txt", "w")
        self.f.close()
        
    def getObs(self):
        self.front_car = None
        lane = self.ego.getState()[0]
        obs = [lane/4, ((self.ego.getState()[2] - self.ego.getState()[3])/(self.max_speed-self.min_speed)) + 0.5, 300, 300, 300, 300]
        
        if lane == 1:
            obs[4] = 0
        elif lane == 4:
            obs[5] = 0
            
        for elem in self.elements:
            if isinstance(elem, Car):
                interdist = elem.getState()[1] - self.ego.getState()[1]
                diff_vel = elem.getState()[2] - self.ego.getState()[2]
                if interdist*diff_vel <= 0:
                    coll_time = (abs(interdist) + 1)/(abs(diff_vel) + 1)
                else:
                    coll_time = abs(interdist) + 1
                    
                if elem.getState()[0] == lane and interdist > 0:
                    if obs[2] > coll_time:
                        obs[2] = coll_time
                elif elem.getState()[0] == lane and interdist < 0:
                    if obs[3] > coll_time:
                        obs[3] = coll_time
                elif elem.getState()[0] == lane + 1:
                    if obs[5] > coll_time:
                        obs[5] = coll_time
                elif elem.getState()[0] == lane - 1:
                    if obs[4] > coll_time:
                        obs[4] = coll_time
                        
                #coll_time = 2*coll_time
        
        obs[2] = np.log(obs[2]+1)/np.log(301)
        obs[3] = np.log(obs[3]+1)/np.log(301)
        obs[4] = np.log(obs[4]+1)/np.log(301)
        obs[5] = np.log(obs[5]+1)/np.log(301)
        
        # obs[2] = (obs[2]+1)/301
        # obs[3] = (obs[3]+1)/301
        # obs[4] = (obs[4]+1)/301
        # obs[5] = (obs[5]+1)/301
                    
        return obs
                
#     def setFree(self):
#         for e in self.elements:
#             if isinstance(e, Car) or isinstance(e, Ego):
#                 e.setLeftFree(True)
#                 e.setRightFree(True)
#                 for elem in self.elements:
#                     interdist = elem.getState()[1] - e.getState()[1]
#                     diff_vel = elem.getState()[2] - e.getState()[2]
#                     if interdist*diff_vel <= 0:
#                         coll_time = (abs(interdist))/(abs(diff_vel) + 1)
#                     else:
#                         coll_time = abs(interdist)

#                     if elem.getState()[0] == e.getState()[0] + 1 or e.getState()[0] == 4:
#                         if coll_time < 50:
#                             e.setLeftFree(False)
#                     elif elem.getState()[0] == e.getState()[0] - 1 or e.getState()[0] == 1:
#                         if coll_time < 50:
#                             e.setRightFree(False)
#                 if e.getState()[0] == 4:
#                     e.setLeftFree(False)
#                 elif e.getState()[0] == 1:
#                     e.setRightFree(False)
        
        
    def reset(self):
        # Reset the reward
        self.ep_return  = 0
        self.t = 0
        # Initialise the ego vehicle
        self.ego = Ego()
        
        self.car_count = 0
        
        # Initialise the elements 
        self.elements = [self.ego]
        
        self.obs = self.getObs()
        return self.obs
        
    def get_action_meanings(self):
        return {0: "Left", 1: "Right", 2: "Maintain"}
    
    def has_collided(self, elem1, elem2):
        if elem1.getState()[0] == elem2.getState()[0]:
            if abs(elem1.getState()[1] - elem2.getState()[1]) < 2:
                return True
        return False
        
    
    def step(self, action):
        t = 1
        self.t = self.t + 1
        # Flag that marks the termination of an episode
        done = False
        
        reward = 1
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"
        
        if action == 0:
            self.a = "left"
        elif action == 1:
            self.a = "right"
        else:
            self.a = "maintain"
            
        self.actions.append(self.a)
        count = [0, 0, 0, 0]
        # Set front car for every car
        for elem in self.elements:
            elem.front_car = None
            dist = 1000
            for e in self.elements:
                if e != elem:
                    if e.getState()[0] == elem.getState()[0]:
                        if e.getState()[1] > elem.getState()[1]:
                            if e.getState()[1] < dist:
                                dist = e.getState()[1]
                                elem.setFrontCar(e)
                                
            count[elem.getState()[0] - 1] += 1

        
        # Apply the action to the ego
        if action == 0:
            if self.ego.getState()[0] == 4:
                reward -= 1
                self.ego.adapt()
            else:
                self.ego.left()
                # self.ego.adapt()
        elif action == 1:
            if self.ego.getState()[0] == 1:
                reward -= 1
                self.ego.adapt()
            else:
                self.ego.right()
                #self.ego.adapt()
        elif action == 2:
            self.ego.adapt()
            reward += 1 # Maintain it's good
        
        self.obs = self.getObs()
        reward += self.obs[2]
        if self.ego.front_car != None and self.obs[2] < 0.7:
            if self.ego.front_car.getState()[2] - self.ego.target < 0:
                if self.obs[5] > 0.8:
                    if action == 0 and (self.obs[5] > self.obs[2] or self.obs[5] > self.obs[3]):
                        reward += 1
                    else:
                        reward -= 1
                else:
                    if action == 2:
                        reward += 1
                    else:
                        reward -= 1
        
        if self.obs[3] < 0.7 or self.ego.getState()[2] - self.ego.target == 0:
            if self.obs[4] > 0.8:
                if action == 1 and (self.obs[4] > self.obs[2] or self.obs[4] > self.obs[3]):
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 2:
                    reward += 1
                else:
                    reward -= 1
        # reward += 4-4*self.obs[0]
        # reward -= 10*abs(self.obs[1] - 0.5) # Penalize if the velocity is not the target
        self.observations.append(self.obs)
        
        # For elements in the Ev
        for elem in self.elements:
            if isinstance(elem, Car):
                elem.tick(t, self.ego.getState()[2], action)
                if elem.getState()[1] < -300 or elem.getState()[1] > 300:
                    self.elements.remove(elem)
                    self.car_count = self.car_count - 1
                
                # If the car has collided.
                if self.has_collided(self.ego, elem):
                    # Conclude the episode and remove the chopper from the Env.
                    done = True
                    self.f = open("history.txt", "a")
                    self.f.write("Crashed\n")
                    self.f.write("Crashing car: " + str(elem.getState()))
                    self.f.write("\n")
                    self.f.write("Ego: " +str(self.ego.getState()))
                    self.f.write("\n")
                    if not(self.ego.front_car == None):
                        self.f.write("Front: " + str(self.ego.front_car.getState()))
                        self.f.write("\n")
                    for i in range(1, min(10, self.t - 1)):
                        self.f.write("Time: " + str(self.t - i))
                        self.f.write("\n")
                        self.f.write(str(self.observations[self.t - i]))
                        self.f.write("\n")
                        self.f.write("Action taken: " + str(self.actions[self.t - i]))
                        self.f.write("\n")
                    self.f.write("++++++++++++++++++++++++++++++++++++++++\n")
                    self.f.close()
                    reward -= 1000
                    #self.printEnv(self.a)
                    time.sleep(2)
                    # self.elements.remove(self.ego)
            else:
                elem.tick(t)
                    
                    
        if self.car_count < 10 and random.randint(1,10) < 2:
            new_car = Car()
            self.elements.append(new_car)
            self.car_count = self.car_count + 1

        # Increment the episodic return
        self.ep_return += 1
        
        if self.t > 300:
            done = True
            self.f = open("history.txt", "a")
            self.f.write("Well done\n")
            self.f.write("Final distance: " + str(self.ego.getDistance()))
            self.f.write("\n")
            self.f.write("---------------------------------------\n")
            self.f.close()
            
    
        return self.obs, reward, done, []
    
    def printEnv(self, action):
        fig, ax = plt.subplots()
        for elem in self.elements:
            if isinstance(elem, Ego):
                ax.plot(elem.getState()[1], elem.getState()[0], 'ro')
            else:
                ax.plot(elem.getState()[1], elem.getState()[0], 'bo')
        ax.set_ylim(0, 5)
        ax.set_xlim(-300, 300)
        plt.show()
        print(self.a)
        print(self.obs)
        print(self.ego.getState()[2])
        time.sleep(0.001)
        
    def getDist(self):
        return self.ego.distance
    

class Car(object):
    def __init__(self):
        self.lane = random.randint(1,4)
        if random.randint(1,3) < 2:
            self.position = 200
        else:
            self.position = -200
        
        if self.lane == 1:
            self.velocity = random.randint(15,18)
            self.target = random.randint(15,20)
        elif self.lane == 2:
            self.velocity = random.randint(18,20)
            self.target = random.randint(15,23)
        elif self.lane == 3:
            self.velocity = random.randint(20,25)
            self.target = random.randint(18,25)
        elif self.lane == 4:
            self.velocity = random.randint(23,25)
            self.target = random.randint(20,25)
        
    def getState(self):
        return np.array([self.lane, self.position, self.velocity, self.target])
    
    def accelerate(self):
        if self.velocity < 25:
            self.velocity += 1
        
    def brake(self):
        if self.velocity > 15:
            self.velocity -= 1
            
    def setFrontCar(self, front):
        self.front_car = front
            
    def adapt(self):
        if self.velocity > self.target:
            self.brake()
        elif self.front_car != None:
            if self.velocity < self.target:
                if self.velocity < self.front_car.getState()[2]:
                    self.accelerate()
                elif (self.front_car.getState()[1] - self.position)> 2*self.velocity:#/(self.velocity - self.front_car.getState()[2] + 1) > self.velocity:
                    self.accelerate()
                elif (self.front_car.getState()[1] - self.position)< 2*self.velocity:#/(self.velocity - self.front_car.getState()[2] + 1) < self.velocity:
                    self.brake()
            else:
                if (self.front_car.getState()[1] - self.position) < 2*self.velocity: #/(self.velocity - self.front_car.getState()[2] + 1) < self.velocity:
                    self.brake()
        elif self.velocity < self.target:
            self.accelerate()
        
    def tick(self, t, vel, a):
        self.adapt()
        self.position = self.position - (vel - self.velocity)*t
        
        
class Ego(object):
    def __init__(self):
        self.lane = random.randint(1,4)
        self.position = 0
        self.velocity = random.randint(15,25)
        self.distance = 0
        self.front_car = None
        self.target = 21
        
    def getState(self):
        return np.array([self.lane, self.position, self.velocity, self.target])
        
    def left(self):
        if self.lane < 4:
            self.lane += 1
        
    def right(self):
        if self.lane > 1:
            self.lane -= 1
        
    def accelerate(self):
        if self.velocity < 25:
            self.velocity += 1
        
    def brake(self):
        if self.velocity > 15:
            self.velocity -= 1
            
    def getDistance(self):
        return self.distance
    
    def setFrontCar(self, front):
        self.front_car = front
        
    def adapt(self):
        if self.velocity > self.target:
            self.brake()
        elif self.front_car != None:
            if self.velocity < self.target:
                if self.velocity < self.front_car.getState()[2]:
                    self.accelerate()
                elif (self.front_car.getState()[1] - self.position)> 2*self.velocity:#/(self.velocity - self.front_car.getState()[2] + 1) > self.velocity:
                    self.accelerate()
                elif (self.front_car.getState()[1] - self.position)< 2*self.velocity:#/(self.velocity - self.front_car.getState()[2] + 1) < self.velocity:
                    self.brake()
            else:
                if (self.front_car.getState()[1] - self.position)< 2*self.velocity:#/(self.velocity - self.front_car.getState()[2] + 1) < self.velocity:
                    self.brake()
        elif self.velocity < self.target:
            self.accelerate()
    
    def tick(self, t):
        self.distance = self.distance + self.velocity*t