from sprites import *
import torch
import graphics as D
import torch


class Environment:
    def __init__(self) -> None:
        self.car = Car(2)
        self.obstacles_group = pygame.sprite.Group()
        self.good_points_group= pygame.sprite.Group()
        #self.spawn_timer = 0
        self.score=0
        GoodPoint.indecis = [None] * 5
        self.coin_reward = 0.5
        self.lose_reward = -1
        self.change_line_reward = 0
        self.i_reward = 0.03
        

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
        # pointCollided=pygame.sprite.spritecollide(self.car,self.good_points_group,True)
        # if len(pointCollided) != 0:
        #     self.score+=1
        # Custom collision detection for coins
        if len(pygame.sprite.spritecollide(self.car,self.good_points_group,True)) !=0:
             self.score += 1  # Increment the score
             self.reward+=self.coin_reward
        # for sprite in self.good_points_group:
        #     rect = sprite.rect

    def reset(self):#for AI, we dont need screen,  print is good enough.
        from game import game
        print(self.score)
        game.loop()

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

    def update (self,action):
        self.reward=0
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
        
        for obstacle in self.obstacles_group:
            if obstacle.rect.top > 800 :
                obstacle.kill()
                self.obstacles_group.remove(obstacle)
        for GoodPoint in self.good_points_group:
            if GoodPoint.rect.top > 800 :
                GoodPoint.kill()
                self.good_points_group.remove(GoodPoint)
        return (False,self.reward)
        
                 
    def first_sprite_in_lane(self, state):
        car_lane_onehot = state[:5]  # shape: (5,)
        
        # --- Slice obstacles and coins ---
        obstacles = state[5:5 + 4 * 6].view(4, 6)  # shape: (4, 6)
        coins     = state[5 + 4 * 6:].view(5, 6)    # shape: (5, 6)
        
        # --- Extract lane one-hot vectors and y-values ---
        obstacle_lanes = obstacles[:, :5]         # shape: (4, 5)
        obstacle_ys    = obstacles[:, 5]            # shape: (4,)
        
        coin_lanes = coins[:, :5]                 # shape: (5, 5)
        coin_ys    = coins[:, 5]                  # shape: (5,)
        
        # Clamp y-values to a minimum of 0 so that negative y (coming in) becomes 0.
        obstacle_ys = obstacle_ys.clamp(min=0)
        coin_ys = coin_ys.clamp(min=0)
        
        # --- Mask: which objects are in the same lane as the car ---
        same_lane_obstacles = (obstacle_lanes == car_lane_onehot).all(dim=1)  # shape: (4,)
        same_lane_coins     = (coin_lanes == car_lane_onehot).all(dim=1)      # shape: (5,)
        
        # --- Filter y-values for objects in the same lane ---
        obstacle_y_filtered = obstacle_ys[same_lane_obstacles]
        coin_y_filtered     = coin_ys[same_lane_coins]
        
        # --- Find the sprite with maximum y (closest to the car) ---
        found = False
        closest_y = 0.0
        closest_type = None

        if obstacle_y_filtered.numel() > 0:
            candidate = obstacle_y_filtered.max().item()
            closest_y = candidate
            closest_type = -1  # obstacle
            found = True

        if coin_y_filtered.numel() > 0:
            candidate = coin_y_filtered.max().item()
            if (not found) or (candidate >= closest_y):
                closest_y = candidate
                closest_type = 1  # coin
                found = True

        if not found:
            return None, 0.0
        else:
            return closest_type, closest_y

        
    def lane_to_one_hot (self, lane):
        lane_lst = [0] * 5
        lane_lst[lane] = 1
        return lane_lst

    def one_hot_to_lane (self, lane_lst):
        return lane_lst.index(1)

    def immediate_reward(self, state, next_state):
        # Extract lane information 
        lane1 = self.one_hot_to_lane(state[:5].tolist())
        lane2 = self.one_hot_to_lane(next_state[:5].tolist())
    
        # Get sprite type and metric from the lane.
        type1, max_y1 = self.first_sprite_in_lane(state)
        type2, max_y2 = self.first_sprite_in_lane(next_state)
        
        # Ensure positions are non-negative.
        max_y1 = max(0, max_y1)
        max_y2 = max(0, max_y2)
        
        reward = 0
        
        # If there's no sprite in either state, no immediate reward.
        if type1 is None and type2 is None:
            return reward

        # Case 1: Staying in a coin lane.
        if type1 == 1 and type2 == 1:
            reward = self.i_reward * max_y2#(max_y2-max_y1)

        # Case 2: Staying in an obstacle lane.
        elif type1 == -1 and type2 == -1:
            reward = -self.i_reward * max_y2#(max_y2-max_y1)

        # Case 3: Transition from obstacle to coin.
        elif type1 == -1 and type2 == 1:
            reward = self.i_reward * (max_y2 + max_y1)
        
        # Case 4: Transition from coin to obstacle.
        elif type1 == 1 and type2 == -1:
            if lane1 == lane2:  # Car stayed in lane: coin was collected.
                reward += 0   #reward given elsewhere 
            else:
                reward = -self.i_reward * (max_y2 + max_y1)

        # Case 5: Transition from coin to clear.
        elif type1 == 1 and type2 is None:
            if lane1 == lane2: # Collected coin
                reward += 0  # reward given elsewhere.
            else:
                reward = -self.i_reward * max_y1  # Left coin behind.

        # Case 6: Transition from obstacle to clear.
        elif type1 == -1 and type2 is None:
            reward = self.i_reward * max_y1

        # Case 7: Transition from clear to coin.
        elif type1 is None and type2 == 1:
            reward = self.i_reward * max_y2

        # Case 8: Transition from clear to obstacle.
        elif type1 is None and type2 == -1:
            reward = -self.i_reward * max_y2

        return reward
    

    def new_game(self):
        self.car.lane = 2
        self.obstacles_group = pygame.sprite.Group()
        self.good_points_group= pygame.sprite.Group()
        self.score=0
        GoodPoint.indecis = [None] * 5