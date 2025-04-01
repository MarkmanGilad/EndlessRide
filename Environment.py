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
        state_list.append((self.car.lane+1))  # Add the car's lane 1-5

        # 2. Obstacle Positions
        for obstacle in self.obstacles_group:
            state_list.append((obstacle.lane+1))  # X-coordinate of obstacle
            state_list.append(obstacle.rect.y/700)  # Y-coordinate of obstacle
        while (len(state_list)<9):
            state_list.append(0)  
            state_list.append(0)  
        # 3. Good Point Positions
        for good_point in GoodPoint.indecis:
            if good_point:
                state_list.append(good_point.lane+1)  # X-coordinate of good point
                state_list.append(good_point.rect.y/700)  # Y-coordinate of good point
            else:   
                state_list.append(0)  
                state_list.append(0)  

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
        
                 
    def first_sprite_in_lane (self, state):
        data = state[1:].view(-1, 2)
        filtered = data[data[:, 0] == state[0].item()]
        if len(filtered) > 0:
            max_y, idx = filtered[:, 1].max(dim=0)
            idx = idx.item() 
            max_y = max_y.item()
            if idx < 4:
                type = -1 # obsticales
            else:
                type = 1 # coins
            return type, max_y
        else:
            return None, 0
        
    def imediate_reward(self, state, next_state):
        type1, max_y1 = self.first_sprite_in_lane(state)
        type2, max_y2 = self.first_sprite_in_lane(next_state)
        reward = 0
        max_y1 = max(0, max_y1)
        max_y2 = max(0, max_y2)
        if not type1 and not type2:
            return reward
        
        if type1 == -1 and type2 == 1:
            reward = self.coin_reward * max_y2
        elif type1 == 1 and type2 == -1:
            reward = self.lose_reward * max_y2
        elif type1 == 1 and type2 == 1:
            reward = self.coin_reward * max_y2-max_y1
        elif type1 == -1 and type2 == -1:
            reward = self.lose_reward * max_y2 - max_y1
        elif type1 == 1 and type2 == None:
            reward = -self.coin_reward * max_y1
        elif type1 == -1 and type2 == None:
            reward = -self.lose_reward * max_y1
        elif type1 == None and type2 == 1:
            reward = self.coin_reward * max_y2
        elif type1 == None and type2 == -1:
            reward = self.lose_reward * max_y2
        return reward
        