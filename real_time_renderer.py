import time
from omni.isaac.core.simulation_context import SimulationContext

def render (world: SimulationContext, dt: float, simulation_seconds: float):
  while True:
    before_time = time.time()
    world.reset()
    for j in range(round(simulation_seconds / dt)):
      world.step(render=True)
      wake_time = time.time() - before_time
      if wake_time < dt:
        time.sleep(dt - wake_time)
      before_time = time.time()
