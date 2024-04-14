import os
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cloner import Cloner
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.articulations import ArticulationView
import numpy as np
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core import World
from typing import TypedDict

class Env (TypedDict):
  world: SimulationContext
  cartpoles: ArticulationView
  physics_dt: float
  rendering_dt: float
  xn: int
  yn: int

class CartpoleOriginPositionsGenerator:
  def __init__ (self, xn: int, yn: int):
    self.xn = xn
    self.yn = yn
  
  def generate (self):
    ret: np.ndarray = np.zeros((self.xn * self.yn, 3), dtype=np.float32)
    for y in range(self.yn):
      for x in range(self.xn):
        ret[y * self.xn + x][0] = x
        ret[y * self.xn + x][1] = 7 * y
    
    return ret

def generate_cartpoles (cartpole_origin_positions: np.ndarray[np.ndarray[np.float32]]):
  assets_root_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/'
  cartpole_usd_path = os.path.join(assets_root_path, 'Isaac/Robots/Cartpole/cartpole.usd')
  add_reference_to_stage(cartpole_usd_path, '/World/Cartpole_0')

  Articulation(prim_path='/World/Cartpole_0', name='cartpole')

  cloner = Cloner()
  cloner.filter_collisions('/physicsScene', '/World', list(map(lambda x: '/World/Cartpole_{}'.format(x), list(range(0, cartpole_origin_positions.shape[0])))))
  target_paths = cloner.generate_paths("/World/Cartpole", cartpole_origin_positions.shape[0])
  cloner.clone(source_prim_path="/World/Cartpole_0", prim_paths=target_paths, positions=cartpole_origin_positions)
  cartpoles = ArticulationView('/World/Cartpole_*', 'cartpole_view')

  return cartpoles

def initialize_env (world: SimulationContext, cartpoles: ArticulationView, physics_dt: float, rendering_dt: float):
  world.set_simulation_dt(physics_dt, rendering_dt)
  world.scene.add(cartpoles)
