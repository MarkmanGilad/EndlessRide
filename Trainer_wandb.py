import pygame
import sprites
from graphics import Background
import random
from Environment import Environment

from ReplayBuffer_n_step import ReplayBuffer_n_step as ReplayBuffer
from AI_Agent import AI_Agent
from CNN_DQN import Duelimg_CNN_DQN as DQN
import torch
import wandb
import os
def main (chck):

    pygame.init()
    
    #CONSTS
    FPS = 60
    WINDOWWIDTH = 400
    WINDOWHEIGHT = 800
    MIN_BUFFER=500
    MODEL_PATH = "model/DQN.pth"  # Ensure cross-platform path
    clock = pygame.time.Clock()
    background = Background(WINDOWWIDTH, WINDOWHEIGHT) 
    env = Environment()
    background.render(env)
    best_score = 0
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA")
    else:
        device = torch.device('cpu')
        print("CPU")
    
    ####### params and models ############
    dqn_model = DQN(device=device)
    # dqn_model.load_params(MODEL_PATH)
    print("Model loaded successfully!")
    player = AI_Agent(dqn_model,device=device)
    player_hat = AI_Agent(dqn_model,device=device)
    player_hat.dqn_model = player.dqn_model.copy()
    batch_size = 128
    buffer = ReplayBuffer(path=None)
    learning_rate = 0.001
    ephocs = 200000
    start_epoch = 0
    C = 10
    loss = torch.tensor(0)
    avg = 0

    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player.dqn_model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[5000, 10000, 20000], gamma=0.5)
    step = 0

    #region######## checkpoint Load ############
    num = chck
    checkpoint_path = f"Data/checkpoint{num}.pth"
    buffer_path = f"Data/buffer{num}.pth"
    resume_wandb = False
    if os.path.exists(checkpoint_path):
        resume_wandb = True
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player.dqn_model.load_state_dict(checkpoint['model_state_dict'])
        player_hat.dqn_model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        buffer = torch.load(buffer_path)
        losses = checkpoint['loss']
        scores = checkpoint['scores']
        avg_score = checkpoint['avg_score']
    player.dqn_model.train()
    player_hat.dqn_model.eval()
    #endregion
    #region################ Wandb.init #####################
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="Endless_Road",
        resume=resume_wandb, 
        id=f'Endless_Road {num}',
        # track hyperparameters and run metadata
        config={
        "name": f"Endless_Road {num}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": ephocs,
        "start_epoch": start_epoch,
        "decay": 50,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C,
        "Model":str(player.dqn_model),
        "device": str(device)
        }
    )
    wandb.config.update({"Model":str(player.dqn_model)}, allow_val_change=True)
    #endregion
        

    for epoch in range(start_epoch, ephocs):
        step = 0
        #clock = pygame.time.Clock()
        env.new_game()
        background.render(env)

        end_of_game = False
        state = env.state()
        
        while not end_of_game:
            step += 1
            pygame.event.pump()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': player.dqn_model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': losses,
                        'scores':scores,
                        'avg_score': avg_score
                    }
                    torch.save(checkpoint, checkpoint_path)
                    torch.save(buffer, buffer_path)
                    return
            
            ############## Sample Environement #########################
            action = player.getAction(state=state, epoch=epoch)
            done,reward = env.update(action)
            next_state = env.state()
            imediate_reward = env.immediate_reward (state, next_state)
            # immediate_reward = 0
            reward += imediate_reward
            buffer.push(state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), 
                        next_state, torch.tensor(done, dtype=torch.float32))
            if done:
                best_score = max(best_score, env.score)
                break
            else:
                background.render(env)

            state = next_state
            pygame.display.flip()
            
            if len(buffer) < MIN_BUFFER:
                continue
    
            ############## Train ################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = player.Q(states, actions)

            next_actions, _ = player.get_Actions_Values(next_states)
            Q_hat_Values = player_hat.Q(next_states,next_actions) # DDQN
            
            # _, Q_hat_Values = player_hat.get_Actions_Values(next_states) # DQN


            loss = player.dqn_model.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(player.dqn_model.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad()
        scheduler.step()

        if epoch % C == 0:
            player_hat.fix_update(dqn=player.dqn_model)
            

        #region log & print########################################
        print (f'chkpt: {num} epoch: {epoch} loss: {loss:.7f} LR: {scheduler.get_last_lr()}  ' \
               f'score: {env.score} step {step} ')
        
        if epoch % 10 == 0:
            scores.append(env.score)
            losses.append(loss.item())
        wandb.log ({
                "score": env.score,
                "loss": loss.item(),
                "step":step
            })
        step = 0
        avg = (avg * (epoch % 10) + env.score) / (epoch % 10 + 1)
        if (epoch + 1) % 10 == 0:
            avg_score.append(avg)
            wandb.log ({
                # "score": env.score,
                # "loss": loss.item(),
                "avg_score": avg
            })
            print (f'average score last 10 games: {avg} ')
            avg = 0

        if epoch % 10000 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player.dqn_model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': losses,
                'scores':scores,
                'avg_score': avg_score
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)
        #endregion




        
if __name__ == "__main__":
    if not os.path.exists("Data/checkpoit_num"):
        torch.save(101, "Data/checkpoit_num")    
    
    chck = torch.load("Data/checkpoit_num")
    chck += 1
    torch.save(chck, "Data/checkpoit_num")    
    main (chck)