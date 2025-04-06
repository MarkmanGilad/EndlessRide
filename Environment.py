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
        self.coin_reward = 5
        self.lose_reward = -10
        self.change_line_reward = 0
        self.i_reward = 0.02
        self.chkpt = chkpt
        self.car_top_row = 118

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

    def state2D(self):
        # (channels, rows, cols) = (car, obstacle, coin), 140 rows of 5 px, 5 lanes
        grid = torch.zeros((3, 140, 5), dtype=torch.float32)

        # --- 1. Car (always in bounds, occupies 100px = 20 rows) ---
        car_lane = self.car.lane
        car_y = self.car.rect.y   # top of the car 590
        car_top = int(car_y / 5)           # = 118
        car_bottom = int((car_y + 100) // 5)  # = 138
        grid[0, car_top:car_bottom, car_lane] = 1.0  # No +1 because upper bound is exclusive

        # --- 2. Obstacles ---
        for obstacle in self.obstacles_group:
            lane = obstacle.lane
            y = obstacle.rect.y 

            top_row = max(int(y / 5), 0)
            bottom_row = min(int((y + 50) / 5), 140)

            if top_row < bottom_row:
                grid[1, top_row:bottom_row, lane] = 1.0

        # --- 3. Good Points (Coins) ---
        for good_point in self.good_points_group:
                lane = good_point.lane
                y = good_point.rect.y 
                top_row = max(int(y / 5), 0)
                bottom_row = min(int((y + 50) / 5), 140)
                if top_row < bottom_row:
                    grid[2, top_row:bottom_row, lane] = 1.0

        return grid


    def update (self,action):
        self.reward=0
        self.score +=0.1
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
        
                 
    def first_sprite_in_lane(self, state: torch.Tensor):
        car_top_row = self.car_top_row
        # Unwrap batch dimension: [1, 3, 140, 5] → [3, 140, 5]
        if state.dim() == 4:
            state = state.squeeze(0)
        
        # Channels
        CAR, OBSTACLE, COIN = 0, 1, 2

        # 1. Find car's lane from row 130
        car_row = state[CAR, car_top_row, :]  # shape: (5,)
        car_lane = torch.nonzero(car_row).item()  # lane index (0–4)

        # 2. Extract the obstacle and coin column (up to row 130)
        obs_col = state[OBSTACLE, :car_top_row, car_lane]  # shape: (130,)
        coin_col = state[COIN, :car_top_row, car_lane]     # shape: (130,)

        # 3. Find row indices where object exists
        obs_rows = torch.nonzero(obs_col, as_tuple=False).squeeze()
        coin_rows = torch.nonzero(coin_col, as_tuple=False).squeeze()

        # 4. Get bottom row (max row index) for each object type
        obs_bottom = obs_rows.max().item() if obs_rows.numel() > 0 else -1
        coin_bottom = coin_rows.max().item() if coin_rows.numel() > 0 else -1

        # 5. Determine first visible object (closest to car)
        if obs_bottom == -1 and coin_bottom == -1:
            return None, 0  # no object found

        if obs_bottom > coin_bottom:
            return obs_bottom, -1  # obstacle is closer
        else:
            return coin_bottom, 1  # coin is closer

        
    def lane_to_one_hot (self, lane):
        lane_lst = [0] * 5
        lane_lst[lane] = 1
        return lane_lst

    def one_hot_to_lane (self, lane_lst):
        return lane_lst.index(1)

    def immediate_reward(self, state, next_state):
        # Unwrap batch dimension: [1, 3, 140, 5] → [3, 140, 5]
        if state.dim() == 4:
            state = state.squeeze(0)
        if next_state.dim() == 4:
            next_state = next_state.squeeze(0)

        # Extract lane information 
        car1_row = state[0, self.car_top_row, :]  # shape: (5,)
        lane1 = torch.nonzero(car1_row).item()  # lane index (0–4)
        
        car2_row = next_state[0, self.car_top_row, :]  # shape: (5,)
        lane2 = torch.nonzero(car2_row).item()  # lane index (0–4)
        
    
        # Get sprite type and metric from the lane.
        max_y1, type1  = self.first_sprite_in_lane(state)
        max_y2, type2  = self.first_sprite_in_lane(next_state)
                
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
                reward += 0   #reward elsewhere   
            else:
                reward = -self.i_reward * (max_y2 + max_y1)

        # Case 5: Transition from coin to clear.
        elif type1 == 1 and type2 is None:
            if lane1 == lane2: # Collected coin
                reward += 0   #reward elsewhere 
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