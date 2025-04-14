from sprites import *
import torch
import graphics as D
import torch


class Environment:
    def __init__(self, chkpt = 1) -> None:
        self.car = Car(2)
        self.obstacles_group = pygame.sprite.Group()
        self.good_points_group= pygame.sprite.Group()
        #self.spawn_timer = 0
        self.score=0
        GoodPoint.indecis = [None] * 5
        self.coin_reward = 3
        self.lose_reward = -1
        self.change_line_reward = 0
        self.i_reward = 0.5
        self.chkpt = chkpt
        self.car_top_row = 118
        self.car_top = 590

    def move (self, action):
        lane = self.car.lane
        if action == 1 and lane < 4:
            self.car.lane +=1
            
        if action == -1 and lane > 0:
            self.car.lane -=1
        
    def _check_obstacle_placement(self, obstacle):
        collided = pygame.sprite.spritecollide(obstacle, self.obstacles_group, False)
        collided2 = pygame.sprite.spritecollide(obstacle, self.good_points_group, False)
        return len(collided) == 0 and len(collided2) == 0  # Return True if no collisions

    def Max_obstacle_check(self):
        """Checks if there are more than 10 obstacles in the game."""
        if len(self.obstacles_group) >= 4:
            return True  # More than 10 obstacles exist
        else:
            return False # 10 or fewer obstacles exist
            
    def Max_GoodPoints_check(self):
        """Checks if there are more than 10 good points in the game."""
        if len(self.good_points_group) >= 5:
            return True  # More than 5 points exist
        else:
            return False # 5 or fewer points exist
        
    def add_obstacle(self):
        spawn_probability = 0.015  #CHANGE
        if random.random() < spawn_probability:
            obstacle = Obstacle()
            #obstacle.rect.x = random.randrange(0, 400, 80)
            obstacle.rect.y = -obstacle.rect.height  # Spawn at the top of the screen
            if self._check_obstacle_placement(obstacle) and self.Max_obstacle_check() is False:
                self.obstacles_group.add(obstacle)
            else:
                obstacle.kill()

    def add_coins (self):                                                           ###### Gilad
        # Spawn good points (optional)
        spawn_good_point_probability = 0.01 #CHANGE  
        if random.random() < spawn_good_point_probability and len(self.good_points_group) < 5:
            good_point = GoodPoint()
            if self._check_obstacle_placement(good_point):
                self.good_points_group.add(good_point)
            else:
                good_point.kill()

    def car_colide(self) -> bool :
        colides = pygame.sprite.spritecollide(self.car,self.obstacles_group,False)
        return len(colides) ==0

    def AddGood(self):
        if len(pygame.sprite.spritecollide(self.car,self.good_points_group,True)) !=0:
             self.score += 1  # Increment the score
             self.reward+=self.coin_reward
        
    def reset(self):#for AI, we dont need screen,  print is good enough.
        print(self.score)
        # game.loop()

    def state(self):
        state_list = []

        # 1. Car's Lane
        state_list += self.lane_to_one_hot(self.car.lane)  # Add the car's lane 1-5

        # 2. Obstacle Positions
        for obstacle in self.obstacles_group:
            state_list += self.lane_to_one_hot(obstacle.lane)  # X-coordinate of obstacle
            state_list += [obstacle.rect.y/700]  # Y-coordinate of obstacle
        for i in range(4 - len(self.obstacles_group)):
            state_list +=[0]*5
            state_list += [0]
        # 3. Good Point Positions
        for good_point in GoodPoint.indecis:
            if good_point:
                state_list += self.lane_to_one_hot(good_point.lane)  # X-coordinate of good point
                state_list += [good_point.rect.y/700]  # Y-coordinate of good point
            else:   
                state_list += [0]*5
                state_list += [0]

        return torch.tensor(state_list, dtype=torch.float32)
        
    def state_lanes(self):
        
        lane_left = self.lane_encoding(max(self.car.lane-1, 0))  
        lane_stay = self.lane_encoding(self.car.lane)  # Add the car's lane 1-5
        lane_right = self.lane_encoding(min(self.car.lane+1, 4))  
        
        obj_front = [0,0,0,0,0]
        for obstacle in self.obstacles_group:
            lane = obstacle.lane
            y = -max(obstacle.rect.bottom,0)/700
            if abs(obj_front[lane]) < abs(y):
                obj_front[lane] = y

        for coin in self.good_points_group:
            lane = coin.lane
            y = max(coin.rect.bottom,0)/700
            if abs(obj_front[lane]) < abs(y):
                obj_front[lane] = y

        state_lst = lane_left + lane_stay + lane_right + obj_front
        state = torch.tensor(state_lst, dtype=torch.float32)
        state = state.unsqueeze(0) 
        return state

    def update (self,action):
        self.reward=0
        # self.score +=0.1
        prev_lane=self.car.lane
        self.move(action=action)
        if self.car.lane != prev_lane:
            self.reward=self.reward-self.change_line_reward #car change lane reward
        self.add_obstacle()
        self.add_coins()
        
        # Update game objects
        self.car.update()
        self.obstacles_group.update()
        self.good_points_group.update()
        self.AddGood()
        if not self.car_colide():
           return (True,self.lose_reward)  #lose reward
                
        return (False,self.reward)
        
    def lane_to_one_hot (self, lane):
        lane_lst = [0] * 5
        lane_lst[lane] = 1
        return lane_lst

    def lane_encoding (self, lane):
        lane_lst = [0] * 5

        lane_lst[lane] = 5
        for i in range(1, 5):
            if lane - i >= 0:
                lane_lst[lane-i] = 5-i    
            if lane + i <= 4:
                lane_lst[lane+i] = 5-i
        
        return lane_lst
            
        
    def one_hot_to_lane (self, lane_lst):
        return lane_lst.index(1)

    def immediate_reward (self, state, action):
        obj = state[0,15:20]
        lane = self.car.lane
        after_lane = min(max(lane + action,0),4)
        reward_state = obj[lane]
        reward_after_state = obj[after_lane]

        reward = 0
        
        if reward_after_state < 0:
            reward = -self.i_reward
        else:
            reward = self.i_reward

        if action == 0:
            if reward_state < 0:    # Obstacle
                reward = -self.i_reward
            elif reward_state > 0:  # coin
                reward = self.i_reward
            else:                   # empty don't stay on the lane
                reward = -self.i_reward / 10

        else:
            if reward_state > 0 and reward_after_state > 0: # coin -> coin
                if reward_after_state > reward_state:
                    reward = self.i_reward / 5
                else:
                    reward = -self.i_reward / 5
            elif reward_state > 0 and reward_after_state < 0: # coin -> obsticale
                reward = -self.i_reward * 2
            elif reward_state > 0 and reward_after_state == 0: # coin -> empty
                reward = -self.i_reward
            elif reward_state < 0 and reward_after_state < 0: # obsticale -> obsticale
                if reward_after_state > reward_state:
                    reward = self.i_reward / 5
                else:
                    reward = -self.i_reward / 5
            elif reward_state < 0 and reward_after_state > 0: # obsticale -> coin
                reward = self.i_reward * 2
            elif reward_state < 0 and reward_after_state == 0: # obsticale -> empty
                reward = self.i_reward
            elif reward_state == 0 and reward_after_state < 0: # empty -> obsticale
                reward = -self.i_reward
            elif reward_state == 0 and reward_after_state > 0: # empty -> coin
                reward = self.i_reward
            elif reward_state == 0 and reward_after_state == 0: # empty -> empty
                reward = self.i_reward / 10

        return reward

    def new_game(self):
        self.car.lane = 2
        self.obstacles_group = pygame.sprite.Group()
        self.good_points_group= pygame.sprite.Group()
        self.score=0
        GoodPoint.indecis = [None] * 5