
"""Compatibility runtime for the Alpha-daemon repository.

This adapter uses the repository's existing ALPHA_GRID_WORLD.py as the
environment substrate. It deliberately does not pretend that this is the
larger AIRE physical-world runtime: the daemon can therefore run against the
world that is actually present in the repository today.
"""
from __future__ import annotations
import hashlib, json, random
from dataclasses import dataclass
from ALPHA_GRID_WORLD import GridWorld

@dataclass
class Report:
    alive: bool
    action: str
    reward: float
    state_digest: str

class AlphaWorld:
    def __init__(self, seed=0, width=64, height=64):
        self.seed=int(seed)
        self.rng=random.Random(self.seed)
        self.world=GridWorld(width,height,self.seed)
        self.x=width//2
        self.y=height//2
        self.alive=True
        self.time=0
        self.steps=0
        self.last_action="observe"
        self.cumulative_reward=0.0

    def _digest(self):
        h=hashlib.sha256()
        h.update(f"{self.time}:{self.x}:{self.y}".encode())
        for row in self.world.cells:
            for c in row:
                h.update(f"{c.value:.8f}:{int(c.blocked)}".encode())
        return h.hexdigest()

    def _choose(self):
        obs=self.world.observe(self.x,self.y,2)
        # A minimal adaptive heuristic: move toward the locally highest
        # observed unblocked cell, without a hand-authored scenario.
        candidates=[]
        for dx,dy,value,blocked in obs:
            if blocked: continue
            candidates.append((value,dx,dy))
        if not candidates:
            return "wait",0,0
        _,dx,dy=max(candidates,key=lambda z:z[0])
        dx=max(-1,min(1,dx)); dy=max(-1,min(1,dy))
        return f"move_{dx}_{dy}",dx,dy

    def step(self, compute_digest=True):
        before=self.world.cells[self.x][self.y].value
        self.world.step()
        action,dx,dy=self._choose()
        nx=max(0,min(self.world.width-1,self.x+dx))
        ny=max(0,min(self.world.height-1,self.y+dy))
        self.x,self.y=nx,ny
        after=self.world.cells[self.x][self.y].value
        reward=float(after-before)
        self.time+=1
        self.steps+=1
        self.last_action=action
        self.cumulative_reward+=reward
        digest=self._digest() if compute_digest else ""
        return Report(self.alive,action,reward,digest)

def create_alpha_world(seed=0):
    return AlphaWorld(seed=seed)
