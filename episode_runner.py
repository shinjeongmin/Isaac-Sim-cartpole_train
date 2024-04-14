from omni.isaac.core.simulation_context import SimulationContext
from stepper import Stepper
from env.env import Env
import numpy as np
import torch
from model.actor import Actor
from model.critic import Critic
from agent_over_manager import AgentOverManager

class EpisodeRunner:
  def __init__ (self, stepper: Stepper, env: Env, actor: Actor, critic: Critic, agent_over_manager: AgentOverManager, episode_num: int, simulation_seconds: float, device):
    self.device = device
    self.stepper = stepper
    self.env = env
    self.actor = actor
    self.critic = critic
    self.agent_over_manager = agent_over_manager
    self.episode_num = episode_num
    self.simulation_seconds = simulation_seconds

  def run_episode (self):
    cnt = 0
    for _ in range(self.episode_num):
      self.env['world'].reset()
      self.env['cartpoles'].set_joint_positions(positions=np.concatenate([np.zeros((self.env['xn'] * self.env['yn'], 1), dtype=np.float32), np.random.normal(size=(self.env['xn'] * self.env['yn'], 1)) * np.pi / 6], axis=1), joint_indices=[0, 1])
      self.env['cartpoles'].set_joint_velocities(velocities=np.concatenate([np.zeros((self.env['xn'] * self.env['yn'], 1), dtype=np.float32), np.zeros((self.env['xn'] * self.env['yn'], 1), dtype=np.float32)], axis=1), joint_indices=[0, 1])
      self.agent_over_manager.init()
      for _ in range(round(self.simulation_seconds / self.env['physics_dt'])):
        with torch.no_grad():
          joint_positions = self.env['cartpoles'].get_joint_positions(joint_indices=[0, 1])
          joint_velocities = self.env['cartpoles'].get_joint_velocities(joint_indices=[0, 1])
          state = torch.from_numpy(
            np.concatenate([
              joint_positions[:, [0]],
              joint_velocities[:, [0]],
              joint_positions[:, [1]],
              joint_velocities[:, [1]]
            ], axis=1)
          )
          a = self.actor(state.to(self.device))
          if cnt >= 100:
            print('a:', a)
            cnt = 0
          else:
            cnt += 1
        self.stepper.step(a.detach().cpu().numpy())
