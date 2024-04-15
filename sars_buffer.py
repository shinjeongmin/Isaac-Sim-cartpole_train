from typing import TypedDict
import numpy as np

# Sars type -> s_t, a, r, s_tp1, done -> 4 * [float] + 1 * [float] + 1 * [float] + 4 * [float] + 1 *[float] = 11 * [float]

class SarsBuffer:
  data: np.ndarray[np.ndarray[float]]

  def __init__ (self):
    self.data = np.zeros((0, 11), dtype=np.float32)
  
  def sample (self, sample_num: int):
    if self.data.shape[0] < sample_num:
      raise Exception('SarsBuffer.sample failed: self.data.shape[0] < sample_num ({} < {})'.format(self.data.shape[0], sample_num))
    else:
      return self.data[np.random.choice(self.data.shape[0], (sample_num, ), replace=False)]

  def push (self, data: np.ndarray[np.ndarray[float]]):
    self.data = np.concatenate([self.data, data], axis=0)
  
  def pop (self, indices: np.ndarray[int]):
    if indices.shape[0] > self.data.shape[0]:
      raise Exception('SarsBuffer.pop: self.indices.shape[0] > self.data.shape[0] ({} > {})'.format(indices.shape[0], self.data.shape[0]))

    self.data = np.delete(self.data, indices, axis=0)

class SizeLimmitedSarsPushBuffer:
  sars_buffer: SarsBuffer
  size_limit: int

  def __init__ (self, sars_buffer: SarsBuffer, size_limit: int):
    self.sars_buffer = sars_buffer
    self.size_limit = size_limit
  
  def push (self, data: np.ndarray[np.ndarray[float]]):
    self.sars_buffer.push(data)
    if self.sars_buffer.data.shape[0] > self.size_limit:
      indices = np.random.choice(self.sars_buffer.data.shape[0], (self.sars_buffer.data.shape[0] - self.size_limit, ), replace=False)
      self.sars_buffer.pop(indices)
  
  def sample (self, sample_num):
    return self.sars_buffer.sample(sample_num)
