from model.actor import Actor
from model.critic import Critic
from safetensors.torch import load_model, save_model
import os

class TrainCallback:
  def handle (self, actor_critic_loss):
    pass

class TrainMonitor (TrainCallback):
  train_num: int

  def __init__ (self):
    self.train_num = 0

  def handle (self, actor_critic_loss):
    self.train_num += 1
    if self.train_num % 100 == 0:
      print('actor loss, critic_loss: {}, {}'.format(*actor_critic_loss))


class TrainSaver (TrainCallback):
  def __init__ (self, actor: Actor, critic: Critic, actor_target: Actor, critic_target: Critic):
    self.train_num = 0
    self.actor = actor
    self.critic = critic
    self.actor_target = actor_target
    self.critic_target = critic_target

  def handle (self, actor_critic_loss):
    self.train_num += 1
    if self.train_num % 100000 == 0:
      project_path = os.path.join(os.getcwd(), 'custom_standalones/cartpole_train')
      save_model(self.actor, os.path.join(project_path, 'ckpt/actor_{}.safetensors'.format(self.train_num)))
      save_model(self.critic, os.path.join(project_path, 'ckpt/critic_{}.safetensors'.format(self.train_num)))
      save_model(self.actor_target, os.path.join(project_path, 'ckpt/actor_target_{}.safetensors'.format(self.train_num)))
      save_model(self.critic_target, os.path.join(project_path, 'ckpt/critic_target_{}.safetensors'.format(self.train_num)))
      print('saved')

class TrainEvaluator (TrainCallback):
  def handle (self, actor_critic_loss):
    pass
    # print('train evaluator')
