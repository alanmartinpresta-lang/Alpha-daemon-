"""AIRE experimental autonomy layer.

This module adds no world knowledge, language, goals, plans or scripted
behaviours. It supplies only generic mechanisms: low-dimensional sensor
encoding, bounded motor primitives, short episodic memory, deterministic
exploration and temporal-difference learning from changes in the organism's
own homeostatic state.

The action set describes actuator affordances, not desired behaviours.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, math
from typing import Mapping
import numpy as np

from .organism import OrganismState, MotorCommand

AUTONOMY_SCHEMA = "A1.7"

# Generic actuator affordances. Their semantics are physical, not motivational.
ACTION_NAMES = (
    "rest", "move_x+", "move_x-", "move_y+", "move_y-", "move_z+", "intake"
)

@dataclass(frozen=True)
class Experience:
    state_key: str
    action: int
    reward: float
    next_state_key: str
    terminal: bool

@dataclass(frozen=True)
class AutonomyConfig:
    learning_rate: float = 0.08
    discount: float = 0.92
    exploration_initial: float = 0.25
    exploration_min: float = 0.02
    exploration_decay: float = 0.999
    memory_capacity: int = 512
    mutation_noise: float = 0.0  # reserved; no parameter mutation during life
    force_magnitude: float = 18.0
    intake_effort: float = 1.0
    reward_scale: float = 1.0
    uncertainty_bonus: float = 0.18

    def __setstate__(self, state):
        # Migration for pre-V203 checkpoints: preserve the retained episodic
        # memory as the initial autobiographical record. Historical experiments
        # that were never present in the checkpoint are deliberately NOT injected.
        for k,v in state.items():
            object.__setattr__(self,k,v)
        if not getattr(self, "autobiography", ()) and getattr(self, "memory", ()):
            object.__setattr__(self, "autobiography", tuple(self.memory))

    def __post_init__(self):
        vals=(self.learning_rate,self.discount,self.exploration_initial,
              self.exploration_min,self.exploration_decay,self.force_magnitude,
              self.intake_effort,self.reward_scale,self.uncertainty_bonus)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("autonomy configuration must be finite")
        if not (0 < self.learning_rate <= 1 and 0 <= self.discount < 1):
            raise ValueError("invalid learning parameters")
        if not (0 <= self.exploration_min <= self.exploration_initial <= 1):
            raise ValueError("invalid exploration range")
        if not (0 < self.exploration_decay <= 1):
            raise ValueError("invalid exploration decay")
        if self.memory_capacity < 1 or self.force_magnitude < 0 or self.intake_effort < 0 or self.uncertainty_bonus < 0:
            raise ValueError("invalid autonomy configuration")

@dataclass(frozen=True)
class AutonomyState:
    q_values: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    memory: tuple[Experience, ...] = ()
    # Persistent life history: unlike the short episodic buffer, this is never truncated.
    # It stores raw experience records, not experiment conclusions or scripted preferences.
    autobiography: list[Experience] = field(default_factory=list)
    epsilon: float = 0.25
    steps: int = 0
    cumulative_reward: float = 0.0
    last_action: int = 0
    last_state_key: str = ""
    last_reward: float = 0.0
    visit_counts: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self):
        clean={}
        for k,v in self.q_values.items():
            vals=tuple(float(x) for x in v)
            if len(vals)!=len(ACTION_NAMES) or not all(math.isfinite(x) for x in vals):
                raise ValueError("invalid Q table")
            clean[str(k)]=vals
        object.__setattr__(self,"q_values",clean)
        object.__setattr__(self,"memory",tuple(self.memory))
        object.__setattr__(self,"autobiography",list(getattr(self,"autobiography",()) or ()))
        raw_counts = dict(getattr(self, "visit_counts", {}) or {})
        counts = {}
        for k, v in raw_counts.items():
            vals = tuple(int(x) for x in v)
            if len(vals) != len(ACTION_NAMES) or any(x < 0 for x in vals):
                raise ValueError("invalid visit counts")
            counts[str(k)] = vals
        # Backward-compatible reconstruction for A1.6 checkpoints: recover
        # action visitation counts from the retained episodic memory.
        if not counts and self.memory:
            for exp in self.memory:
                arr = list(counts.get(exp.state_key, (0,) * len(ACTION_NAMES)))
                if 0 <= int(exp.action) < len(ACTION_NAMES):
                    arr[int(exp.action)] += 1
                counts[exp.state_key] = tuple(arr)
        object.__setattr__(self, "visit_counts", counts)
        if not (0 <= self.epsilon <= 1) or not math.isfinite(self.epsilon):
            raise ValueError("invalid epsilon")
        if self.steps < 0 or not math.isfinite(self.cumulative_reward):
            raise ValueError("invalid autonomy counters")

def _clip01(x):
    return float(np.clip(float(x),0.0,1.0))

def sensor_vector(org: OrganismState) -> np.ndarray:
    s=org.sensors; f=org.functional
    core=org.physiology.compartments.get("core")
    water=0.0 if core is None else _clip01(core.water/42.0)
    oxygen=0.0 if core is None else _clip01(core.oxygen/0.25)
    atp=0.0 if core is None else _clip01(core.atp/1.0)
    nutrient=0.0 if core is None else _clip01(core.nutrient/0.5)
    return np.array([
        _clip01(s.temperature/400.0), _clip01(s.pressure/200000.0),
        _clip01(s.light/1.0), _clip01(s.oxygen/0.30),
        _clip01(s.water/1.0), _clip01(s.nutrient/1.0),
        _clip01(s.contact), _clip01(s.nociception),
        _clip01(np.linalg.norm(org.velocity)/50.0),
        water, oxygen, atp, nutrient,
        _clip01(f.fatigue), _clip01(f.pain), _clip01(f.balance),
    ],dtype=float)

def discretize_observation(v: np.ndarray) -> tuple[int,...]:
    """Coarse sensory quantisation; it prevents memorising arbitrary coordinates."""
    v=np.asarray(v,dtype=float)
    if v.shape!=(16,) or not np.isfinite(v).all():
        raise ValueError("observation must be a finite 16-vector")
    # Five physiological bins, three environmental bins.
    edges=np.array([0.2,0.4,0.6,0.8])
    out=[]
    for i,x in enumerate(v):
        if i in (0,1,2,3,4,5,6,7,9,10,11,12,13,14,15):
            out.append(int(np.digitize(x,edges)))
        else:
            out.append(int(np.digitize(x,np.array([0.33,0.66]))))
    return tuple(out)

def state_key(org: OrganismState) -> str:
    return ",".join(map(str,discretize_observation(sensor_vector(org))))

def homeostatic_score(org: OrganismState) -> float:
    """Scalar state quality derived only from physiological viability signals."""
    c=org.physiology.compartments.get("core")
    if c is None or not org.alive:
        return -1.0
    water=_clip01(c.water/42.0)
    oxygen=_clip01(c.oxygen/0.25)
    atp=_clip01(c.atp/1.0)
    nutrient=_clip01(c.nutrient/0.5)
    integrity=_clip01(c.integrity)
    fatigue=1.0-_clip01(org.functional.fatigue)
    pain=1.0-_clip01(org.functional.pain)

    # Thermal homeostasis: maximum quality near the nominal core temperature.
    # The reward is deliberately bidirectional: being too cold AND too hot
    # both reduce homeostatic quality. This avoids encoding "more heat is
    # always better" and gives Alpha an endogenous signal to regulate toward
    # a stable thermal zone.
    core_temp=float(org.functional.core_temperature)
    thermal_quality=float(np.exp(-((core_temp-310.15)/8.0)**2))

    return float(np.mean((water,oxygen,atp,nutrient,integrity,fatigue,pain,thermal_quality)))

def homeostatic_reward(before: OrganismState, after: OrganismState, dt: float, *,
                        memory: tuple[Experience, ...] = ()) -> float:
    if dt <= 0: raise ValueError("dt must be > 0")
    # Learning receives only endogenous physiological consequences.
    # Exploration is handled separately by uncertainty in the action selector,
    # so the reward itself does not encode a preference for novelty.
    delta = homeostatic_score(after) - homeostatic_score(before)
    survival_bonus = 0.001 if after.alive else -1.0
    return float(np.clip((delta/dt)*10.0 + survival_bonus, -1.0, 1.0))

def action_command(action: int, config: AutonomyConfig) -> MotorCommand:
    if not 0 <= int(action) < len(ACTION_NAMES):
        raise ValueError("unknown action")
    a=int(action); force=np.zeros(3); intake=0.0; activation=0.0
    if a==1: force[0]=config.force_magnitude; activation=0.5
    elif a==2: force[0]=-config.force_magnitude; activation=0.5
    elif a==3: force[1]=config.force_magnitude; activation=0.5
    elif a==4: force[1]=-config.force_magnitude; activation=0.5
    elif a==5: force[2]=config.force_magnitude; activation=0.5
    elif a==6: intake=config.intake_effort; activation=0.15
    return MotorCommand(force=tuple(force), muscle_activation=activation,
                        joint_torque_scale=0.0, intake_effort=intake)

def _ensure(q, key):
    if key in q: return q[key]
    q[key]=(0.0,)*len(ACTION_NAMES); return q[key]

def choose_action(state: AutonomyState, org: OrganismState, *, seed: int,
                  config: AutonomyConfig) -> tuple[int, AutonomyState]:
    key = state_key(org)
    q = dict(state.q_values)
    vals = np.asarray(_ensure(q, key), float)
    counts = dict(state.visit_counts)
    visits = np.asarray(counts.get(key, (0,) * len(ACTION_NAMES)), dtype=float)
    # Uncertainty-driven exploration: actions with fewer direct trials in the
    # current sensory state receive a bounded bonus. This is not a goal and does
    # not encode which action is desirable; it only keeps unknown affordances
    # experimentally accessible as the agent learns.
    uncertainty = config.uncertainty_bonus / np.sqrt(1.0 + visits)
    scores = vals + uncertainty
    rng = np.random.default_rng(int(seed) + state.steps * 0x9E3779B1)
    if rng.random() < state.epsilon:
        action = int(rng.integers(len(ACTION_NAMES)))
    else:
        max_score = float(np.max(scores))
        candidates = np.flatnonzero(np.isclose(scores, max_score, rtol=0.0, atol=1e-12))
        action = int(candidates[rng.integers(len(candidates))])
    arr = list(counts.get(key, (0,) * len(ACTION_NAMES)))
    arr[action] += 1
    counts[key] = tuple(arr)
    ns = AutonomyState(
        q_values=q, memory=state.memory, autobiography=state.autobiography,
        epsilon=state.epsilon, steps=state.steps,
        cumulative_reward=state.cumulative_reward, last_action=action,
        last_state_key=key, last_reward=state.last_reward, visit_counts=counts
    )
    return action, ns

def learn(state: AutonomyState, before: OrganismState, action: int,
          after: OrganismState, *, dt: float, config: AutonomyConfig) -> AutonomyState:
    s=state_key(before); ns=state_key(after)
    q=dict(state.q_values)
    current=np.asarray(_ensure(q,s),float)
    nxt=np.asarray(_ensure(q,ns),float)
    reward=homeostatic_reward(before, after, dt, memory=state.memory)
    target=reward + (0.0 if not after.alive else config.discount*float(np.max(nxt)))
    current[int(action)] += config.learning_rate*(target-current[int(action)])
    q[s]=tuple(current)
    exp=Experience(s,int(action),reward,ns,not after.alive)
    mem=(state.memory+(exp,))[-config.memory_capacity:]
    autobiography=state.autobiography.copy()
    autobiography.append(exp)
    eps=max(config.exploration_min,state.epsilon*config.exploration_decay)
    return AutonomyState(
        q_values=q,
        memory=mem,
        autobiography=autobiography,
        epsilon=eps,
        steps=state.steps+1,
        cumulative_reward=state.cumulative_reward+reward,
        last_action=int(action),
        last_state_key=ns,
        last_reward=reward,
        visit_counts=dict(state.visit_counts),
    )

def autonomy_digest(state: AutonomyState) -> str:
    h=hashlib.sha256(AUTONOMY_SCHEMA.encode())
    for k in sorted(state.q_values):
        h.update(repr((k,state.q_values[k])).encode())
    h.update(repr((state.epsilon,state.steps,state.cumulative_reward,state.last_action,
                  state.last_state_key,state.last_reward,len(state.memory))).encode())
    for k in sorted(state.visit_counts):
        h.update(repr((k,state.visit_counts[k])).encode())
    # Bounded telemetry hash: hashing the full episodic memory every step makes
    # diagnostics O(n^2) over long autonomous runs. The memory itself remains
    # intact; only its digest uses a fixed-size tail.
    h.update(repr(state.memory[-16:]).encode())
    return h.hexdigest()
