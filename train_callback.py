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
  def handle (self, actor_critic_loss):
    pass
    # print('train saver')

class TrainEvaluator (TrainCallback):
  def handle (self, actor_critic_loss):
    pass
    # print('train evaluator')
