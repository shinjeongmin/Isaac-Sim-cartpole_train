import numpy as np
from omni.isaac.kit import SimulationApp
import os

CONFIG = {
  "headless": False,
}
simulation_app = SimulationApp(launch_config=CONFIG)


from omni.isaac.core import World
from env.env import Env, generate_cartpoles, initialize_env, CartpoleOriginPositionsGenerator

physics_dt = 1.0 / 120.0
rendering_dt = 1.0 / 30.0
xn = 25
yn = 4
assets_root_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/'
cartpole_usd_path = os.path.join(assets_root_path, 'Isaac/Robots/Cartpole/cartpole.usd')

world = World(stage_units_in_meters=1.0)
cartpole_origin_positions = CartpoleOriginPositionsGenerator(xn, yn).generate()
cartpoles = generate_cartpoles(cartpole_origin_positions)
initialize_env(world, cartpoles, physics_dt, rendering_dt)
env: Env = {
  'world': world,
  'cartpoles': cartpoles,
  'cartpole_origin_positions': cartpole_origin_positions,
  'physics_dt': physics_dt,
  'rendering_dt': rendering_dt,
  'xn': xn,
  'yn': yn
}


import torch
from model.actor import Actor
from model.critic import Critic
from utils import hard_update

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actor = Actor().to(device)
critic = Critic().to(device)
target_actor = Actor().to(device)
target_critic = Critic().to(device)
hard_update(target_actor, actor)
hard_update(target_critic, critic)


from sars_buffer import SarsBuffer, SizeLimmitedSarsPushBuffer

sars_buffer = SizeLimmitedSarsPushBuffer(SarsBuffer(), 10000)


from episode_runner import EpisodeRunner
from stepper import Stepper, StepForTrain
from trainer import TrainWithCallback, Trainer
from train_callback import TrainSaver, TrainEvaluator, TrainMonitor
from agent_over_manager import AgentOverManager

agent_over_manager = AgentOverManager(env['xn'] * env['yn'])
episode_runner = EpisodeRunner(
  StepForTrain(
    env['cartpoles'], Stepper(env['world'], env['cartpoles'], True),
    TrainWithCallback(
      Trainer(actor, critic, target_actor, target_critic, sars_buffer, 0.99, device),
      callbacks=[TrainMonitor(), TrainSaver(), TrainEvaluator()]
    ),
    agent_over_manager,
    sars_buffer,
    env['xn'] * env['yn']
  ), env, actor, critic, agent_over_manager, 1000, 5, device
)
episode_runner.run_episode()

simulation_app.close()
