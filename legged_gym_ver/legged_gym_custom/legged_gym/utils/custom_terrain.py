from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import numpy as np
from .terrain import Terrain

class custom_Terrain(Terrain):
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        if cfg.difficulty is not None:
            self.difficulty = cfg.difficulty # This will be a list
            print("Custom terrain is generated")
        else:
            self.difficulty = None        
        super().__init__(cfg, num_robots)

            
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            if self.difficulty is not None:
                difficulty = np.random.choice(self.difficulty)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)