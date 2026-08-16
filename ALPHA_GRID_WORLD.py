
from dataclasses import dataclass
import random

@dataclass
class Cell:
    x: int
    y: int
    value: float = 0.0
    blocked: bool = False

class GridWorld:
    """Minimal external substrate: a changing 2-D grid, no task scenario."""
    def __init__(self, width=64, height=64, seed=0):
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.cells = [[Cell(x,y,self.rng.random()) for y in range(height)]
                      for x in range(width)]
        self.t = 0

    def observe(self, x, y, radius=2):
        obs = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                xx, yy = x+dx, y+dy
                if 0 <= xx < self.width and 0 <= yy < self.height:
                    c = self.cells[xx][yy]
                    obs.append((dx,dy,c.value,c.blocked))
        return obs

    def step(self):
        # The world evolves independently. No puzzle, chase, reward hunt,
        # or hand-authored objective is introduced.
        for _ in range(max(1, (self.width*self.height)//80)):
            x=self.rng.randrange(self.width); y=self.rng.randrange(self.height)
            self.cells[x][y].value = max(0.0,min(1.0,
                self.cells[x][y].value + self.rng.uniform(-0.05,0.05)))
        self.t += 1
