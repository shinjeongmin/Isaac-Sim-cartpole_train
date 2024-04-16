import numpy as np

class AgentOverManager:
  over_table: np.ndarray[np.bool8]

  def __init__ (self, agent_num):
    self.agent_num = agent_num

  def init (self):
    self.over_table = np.zeros((self.agent_num, ), np.bool8)
  
  def update (self, joint_positions: np.ndarray[np.ndarray[np.float32]]):
    over_test = self.over_table | (np.abs(joint_positions[:, 0]) > 2.5) | (np.logical_not((np.abs(joint_positions[:, 1]) <= (np.pi / 2)) | (np.abs(joint_positions[:, 1] - 2 * np.pi) <= (np.pi / 2)) | (np.abs(joint_positions[:, 1] + 2 * np.pi) <= (np.pi / 2))))
    self.over_table[over_test] = True
