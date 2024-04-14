from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView
import numpy as np
from trainer import Trainer
from sars_buffer import SizeLimmitedSarsPushBuffer
from agent_over_manager import AgentOverManager

class Stepper:
  def __init__ (self, world: SimulationContext, cartpoles: ArticulationView, render: bool):
    self.world = world
    self.cartpoles = cartpoles
    self.render = render

  def step (self, a: np.ndarray):
    self.cartpoles.set_joint_efforts(efforts=np.reshape(a, (-1, 1)) * 5, joint_indices=np.repeat(0, a.shape[0]))
    self.world.step(render=self.render)

class StepForTrain (Stepper):
  def __init__ (self, cartpoles: ArticulationView, stepper: Stepper, trainer: Trainer, agent_over_manager: AgentOverManager, sars_buffer: SizeLimmitedSarsPushBuffer, agent_num: int):
    self.cartpoles = cartpoles
    self.stepper = stepper
    self.trainer = trainer
    self.sars_buffer = sars_buffer
    self.agent_num = agent_num
    self.agent_finished = np.zeros((agent_num, 1), dtype=np.int8)
    self.agent_over_manager = agent_over_manager

  def step (self, a: np.ndarray[np.ndarray[float]]):
    prev_joint_positions = self.cartpoles.get_joint_positions(joint_indices=[0, 1])
    prev_joint_velocities = self.cartpoles.get_joint_velocities(joint_indices=[0, 1])
    self.stepper.step(a)
    cur_joint_positions = self.cartpoles.get_joint_positions(joint_indices=[0, 1])
    cur_joint_velocities = self.cartpoles.get_joint_velocities(joint_indices=[0, 1])

    prev_over_table = np.array(self.agent_over_manager.over_table)
    prev_live_table = np.logical_not(prev_over_table)
    self.agent_over_manager.update(cur_joint_positions)
    cur_over_table = np.array(self.agent_over_manager.over_table)

    reward = np.zeros((self.agent_num, 1), dtype=np.float32)
    reward[cur_over_table] = -1.0
    reward[cur_over_table == False] = 0.1

    new_sars = np.concatenate([
      prev_joint_positions[prev_live_table][:, [0]],
      prev_joint_velocities[prev_live_table][:, [0]],
      prev_joint_positions[prev_live_table][:, [1]],
      prev_joint_velocities[prev_live_table][:, [1]],
      a[prev_live_table][:, [0]],
      reward[prev_live_table][:, [0]],
      cur_joint_positions[prev_live_table][:, [0]],
      cur_joint_velocities[prev_live_table][:, [0]],
      cur_joint_positions[prev_live_table][:, [1]],
      cur_joint_velocities[prev_live_table][:, [1]]
    ], axis=1)

    self.sars_buffer.push(new_sars)

    self.trainer.train()
