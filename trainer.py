from model.actor import Actor
from model.critic import Critic
from sars_buffer import SizeLimmitedSarsPushBuffer
from train_callback import TrainCallback
import numpy as np
import torch.nn as nn
import torch
from utils import soft_update

class Trainer:
  def __init__ (self, actor: Actor, critic: Critic, target_actor: Actor, target_critic: Critic, sars_buffer: SizeLimmitedSarsPushBuffer, gamma: float, device):
    self.device = device
    self.actor = actor
    self.critic = critic
    self.target_actor = target_actor
    self.target_critic = target_critic
    self.sars_buffer = sars_buffer
    self.gamma = gamma
    self.critic_criterion = nn.MSELoss(reduction='mean').to(self.device, dtype=torch.float32)
    self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=0.001)
    self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=0.001)
  
  def train (self):
    try:
      sars_data = self.sars_buffer.sample(100)
    except:
      return ([-1234, -1234])

    with torch.no_grad():
      target_next_state = torch.from_numpy(sars_data[:, 6: ]).to(self.device, dtype=torch.float32)
      target_next_action = self.target_actor(target_next_state)
      q_target = torch.from_numpy(sars_data[:, [5]]).to(self.device, dtype=torch.float32) \
        + self.gamma * self.target_critic(torch.cat([target_next_state, target_next_action], dim=1))
    q = self.critic(torch.from_numpy(sars_data[:, 0: 5]).to(self.device, dtype=torch.float32))
    critic_loss = self.critic_criterion(q, q_target)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    actor_loss = -self.critic(
      torch.cat([
        torch.from_numpy(sars_data[:, 0: 4]).to(self.device, dtype=torch.float32),
        self.actor(torch.from_numpy(sars_data[:, 0: 4]).to(self.device, dtype=torch.float32))
      ], dim=1)
    ).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    soft_update(self.target_critic, self.critic, 0.001)
    soft_update(self.target_actor, self.actor, 0.001)

    return (actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy())

class TrainWithCallback (Trainer):
  def __init__ (self, trainer: Trainer, callbacks: list[TrainCallback]):
    self.trainer = trainer
    self.callbacks = callbacks
  
  def train (self):
    actor_critic_loss = self.trainer.train()
    for callback in self.callbacks:
      callback.handle(actor_critic_loss)
