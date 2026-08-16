"""AIRE U0 Universal Core.

U0 establishes a solver-independent representation of a deterministic world:
identity, 3-D space, global time, persistent entities, immutable laws,
events/causality, global state, and a controlled world-step operation.

The core intentionally does not implement chemistry, biology, agents, or
specialized CFD. Existing AIRE physics solvers can be attached above this
layer in later roadmap phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping
import hashlib
import json
import math
import uuid
import numpy as np


_EPS = 1e-15


def _finite_vec3(value: Any, name: str) -> np.ndarray:
    x = np.asarray(value, dtype=float).reshape(-1)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    x = x.copy()
    x.setflags(write=False)
    return x


@dataclass(frozen=True)
class WorldIdentity:
    """Stable identity and lineage metadata for a world instance."""
    world_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_world_id: str | None = None
    seed: int = 0
    schema_version: str = "U0.1"

    def __post_init__(self) -> None:
        if not self.world_id or not isinstance(self.world_id, str):
            raise ValueError("world_id must be a non-empty string")
        if self.parent_world_id == self.world_id:
            raise ValueError("a world cannot be its own parent")
        if int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class Space3D:
    """Cartesian world-space contract."""
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_min: np.ndarray | None = None
    bounds_max: np.ndarray | None = None

    def __post_init__(self) -> None:
        origin = _finite_vec3(self.origin, "origin")
        bmin = None if self.bounds_min is None else _finite_vec3(self.bounds_min, "bounds_min")
        bmax = None if self.bounds_max is None else _finite_vec3(self.bounds_max, "bounds_max")
        if (bmin is None) != (bmax is None):
            raise ValueError("bounds_min and bounds_max must be supplied together")
        if bmin is not None and np.any(bmax <= bmin):
            raise ValueError("bounds_max must be strictly greater than bounds_min")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "bounds_min", bmin)
        object.__setattr__(self, "bounds_max", bmax)

    def contains(self, position: Any) -> bool:
        p = np.asarray(position, dtype=float).reshape(-1)
        if p.shape != (3,) or not np.isfinite(p).all():
            return False
        if self.bounds_min is None:
            return True
        return bool(np.all(p >= self.bounds_min) and np.all(p <= self.bounds_max))

    def to_local(self, position: Any) -> np.ndarray:
        return _finite_vec3(np.asarray(position, dtype=float) - self.origin, "local_position")

    def to_world(self, local_position: Any) -> np.ndarray:
        return _finite_vec3(np.asarray(local_position, dtype=float) + self.origin, "world_position")

    def validate(self) -> None:
        if self.bounds_min is not None and not self.contains(self.origin):
            raise ValueError("space origin must lie inside bounded world space")


@dataclass(frozen=True)
class SimulationClock:
    """Deterministic global simulation clock."""
    time: float = 0.0
    step_index: int = 0
    dt: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time)):
            raise ValueError("time must be finite")
        if int(self.step_index) != self.step_index or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not math.isfinite(float(self.dt)) or self.dt <= 0:
            raise ValueError("dt must be finite and > 0")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "step_index", int(self.step_index))
        object.__setattr__(self, "dt", float(self.dt))

    def advance(self, dt: float | None = None) -> "SimulationClock":
        h = self.dt if dt is None else float(dt)
        if not math.isfinite(h) or h <= 0:
            raise ValueError("step dt must be finite and > 0")
        new_time = self.time + h
        if not math.isfinite(new_time) or new_time <= self.time:
            raise ValueError("clock advance must produce a finite strictly later time")
        return SimulationClock(new_time, self.step_index + 1, self.dt)


@dataclass(frozen=True)
class WorldObject:
    """Persistent world entity; intentionally physics-model agnostic."""
    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = "object"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object_id must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("kind must be a non-empty string")
        object.__setattr__(self, "position", _finite_vec3(self.position, "position"))
        object.__setattr__(self, "velocity", _finite_vec3(self.velocity, "velocity"))
        clean_state = dict(self.state)
        clean_tags = tuple(str(t) for t in self.tags)
        object.__setattr__(self, "state", MappingProxyType(clean_state))
        object.__setattr__(self, "tags", clean_tags)

    def moved(self, position: Any, velocity: Any | None = None, **state_updates: Any) -> "WorldObject":
        state = dict(self.state)
        state.update(state_updates)
        return replace(self, position=position, velocity=self.velocity if velocity is None else velocity, state=state)


@dataclass(frozen=True)
class WorldEvent:
    """Immutable event in the world's event history."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time: float = 0.0
    event_type: str = "state_change"
    source_ids: tuple[str, ...] = ()
    target_ids: tuple[str, ...] = ()
    parent_event_ids: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time)):
            raise ValueError("event time must be finite")
        if not self.event_type:
            raise ValueError("event_type must be non-empty")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "source_ids", tuple(str(x) for x in self.source_ids))
        object.__setattr__(self, "target_ids", tuple(str(x) for x in self.target_ids))
        object.__setattr__(self, "parent_event_ids", tuple(str(x) for x in self.parent_event_ids))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True)
class EventLog:
    events: tuple[WorldEvent, ...] = ()

    def append(self, event: WorldEvent) -> "EventLog":
        if self.events and event.time < self.events[-1].time - _EPS:
            raise ValueError("events must be appended in non-decreasing time order")
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError("duplicate event_id")
        return EventLog(self.events + (event,))

    def since(self, time: float) -> tuple[WorldEvent, ...]:
        return tuple(e for e in self.events if e.time >= float(time))


@dataclass(frozen=True)
class CausalLedger:
    """Explicit directed event-dependency ledger."""
    parents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clean = {str(k): tuple(str(v) for v in vals) for k, vals in self.parents.items()}
        for child, parent_ids in clean.items():
            if child in parent_ids:
                raise ValueError("causal self-loop is not allowed")
        object.__setattr__(self, "parents", MappingProxyType(clean))

    def add(self, event_id: str, parent_event_ids: tuple[str, ...] = ()) -> "CausalLedger":
        event_id = str(event_id)
        parent_event_ids = tuple(str(x) for x in parent_event_ids)
        if event_id in parent_event_ids:
            raise ValueError("causal self-loop is not allowed")
        if event_id in self.parents:
            raise ValueError("event already present in causal ledger")
        data = dict(self.parents)
        data[event_id] = parent_event_ids
        return CausalLedger(data)

    def ancestors(self, event_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(self.parents.get(str(event_id), ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.parents.get(current, ()))
        return tuple(sorted(seen))


@dataclass(frozen=True)
class LawRegistry:
    """Registry for immutable world-law identifiers and parameters."""
    laws: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clean: dict[str, Mapping[str, Any]] = {}
        for key, raw in self.laws.items():
            name = str(key)
            if not name:
                raise ValueError("law name must be non-empty")
            value = dict(raw)
            value.setdefault("version", "1.0")
            value.setdefault("domain", "universal")
            value.setdefault("invariants", ())
            if not isinstance(value["version"], str) or not value["version"]:
                raise ValueError("law version must be a non-empty string")
            if not isinstance(value["domain"], str) or not value["domain"]:
                raise ValueError("law domain must be a non-empty string")
            value["invariants"] = tuple(str(x) for x in value["invariants"])
            clean[name] = MappingProxyType(value)
        object.__setattr__(self, "laws", MappingProxyType(clean))

    def define(self, name: str, parameters: Mapping[str, Any] | None = None, *,
               version: str = "1.0", domain: str = "universal",
               invariants: tuple[str, ...] = ()) -> "LawRegistry":
        name = str(name)
        if not name or name in self.laws:
            raise ValueError("law name must be non-empty and unique")
        if not isinstance(version, str) or not version:
            raise ValueError("law version must be a non-empty string")
        if not isinstance(domain, str) or not domain:
            raise ValueError("law domain must be a non-empty string")
        data = dict(self.laws)
        entry = dict(parameters or {})
        entry.update({"version": version, "domain": domain, "invariants": tuple(invariants)})
        data[name] = entry
        return LawRegistry(data)

    def get(self, name: str) -> Mapping[str, Any]:
        return self.laws[str(name)]


@dataclass(frozen=True)
class WorldState:
    """Canonical immutable global world state."""
    identity: WorldIdentity
    space: Space3D
    clock: SimulationClock
    objects: Mapping[str, WorldObject] = field(default_factory=dict)
    laws: LawRegistry = field(default_factory=LawRegistry)
    events: EventLog = field(default_factory=EventLog)
    causality: CausalLedger = field(default_factory=CausalLedger)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.space.validate()
        objects = {str(k): v for k, v in self.objects.items()}
        for key, obj in objects.items():
            if key != obj.object_id:
                raise ValueError("object mapping key must equal object_id")
            if not np.isfinite(obj.position).all() or not np.isfinite(obj.velocity).all():
                raise ValueError("world objects must have finite kinematics")
        event_ids = set()
        previous_time = -math.inf
        for event in self.events.events:
            if event.event_id in event_ids:
                raise ValueError("duplicate event_id")
            if event.time < previous_time - _EPS or event.time > self.clock.time + _EPS:
                raise ValueError("event time must be ordered and not exceed world time")
            event_ids.add(event.event_id)
            previous_time = event.time
        for event_id, parents in self.causality.parents.items():
            if event_id not in event_ids:
                raise ValueError("causal ledger references an unknown child event")
            if any(parent not in event_ids for parent in parents):
                raise ValueError("causal ledger references an unknown parent event")
        object.__setattr__(self, "objects", MappingProxyType(objects))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def add_object(self, obj: WorldObject) -> "WorldState":
        if obj.object_id in self.objects:
            raise ValueError(f"object already exists: {obj.object_id}")
        data = dict(self.objects)
        data[obj.object_id] = obj
        return replace(self, objects=data)

    def update_object(self, obj: WorldObject) -> "WorldState":
        if obj.object_id not in self.objects:
            raise KeyError(obj.object_id)
        data = dict(self.objects)
        data[obj.object_id] = obj
        return replace(self, objects=data)

    def remove_object(self, object_id: str) -> "WorldState":
        if object_id not in self.objects:
            raise KeyError(object_id)
        data = dict(self.objects)
        del data[object_id]
        return replace(self, objects=data)

    def state_digest(self) -> str:
        payload = {
            "world_id": self.identity.world_id,
            "parent_world_id": self.identity.parent_world_id,
            "seed": self.identity.seed,
            "space": {"origin": self.space.origin.tolist(),
                      "bounds_min": None if self.space.bounds_min is None else self.space.bounds_min.tolist(),
                      "bounds_max": None if self.space.bounds_max is None else self.space.bounds_max.tolist()},
            "clock": {"time": self.clock.time, "step": self.clock.step_index, "dt": self.clock.dt},
            "objects": [
                {
                    "id": o.object_id, "kind": o.kind,
                    "position": o.position.tolist(), "velocity": o.velocity.tolist(),
                    "state": dict(o.state), "tags": list(o.tags),
                }
                for o in sorted(self.objects.values(), key=lambda x: x.object_id)
            ],
            "laws": {k: dict(v) for k, v in sorted(self.laws.laws.items())},
            "causality": {k: list(v) for k, v in sorted(self.causality.parents.items())},
            "metadata": dict(self.metadata),
            "events": [
                {"id": e.event_id, "time": e.time, "type": e.event_type,
                 "source": list(e.source_ids), "target": list(e.target_ids),
                 "parents": list(e.parent_event_ids), "payload": dict(e.payload)}
                for e in self.events.events
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(raw).hexdigest()


class WorldCore:
    """Controlled deterministic world-step engine for U0."""

    def __init__(self, state: WorldState):
        self._state = state

    @property
    def state(self) -> WorldState:
        return self._state

    def add_object(self, obj: WorldObject) -> WorldState:
        self._state = self._state.add_object(obj)
        return self._state

    def add_law(self, name: str, parameters: Mapping[str, Any] | None = None, *,
                version: str = "1.0", domain: str = "universal",
                invariants: tuple[str, ...] = ()) -> WorldState:
        self._state = replace(self._state, laws=self._state.laws.define(
            name, parameters, version=version, domain=domain, invariants=invariants))
        return self._state

    def emit(self, event_type: str, *, source_ids=(), target_ids=(), parent_event_ids=(), payload=None) -> WorldEvent:
        parent_event_ids = tuple(str(x) for x in parent_event_ids)
        known_events = {e.event_id for e in self._state.events.events}
        if any(parent not in known_events for parent in parent_event_ids):
            raise ValueError("all causal parents must already exist in the event history")
        event = WorldEvent(
            time=self._state.clock.time,
            event_type=event_type,
            source_ids=tuple(source_ids), target_ids=tuple(target_ids),
            parent_event_ids=tuple(parent_event_ids), payload=payload or {},
        )
        events = self._state.events.append(event)
        causal = self._state.causality.add(event.event_id, event.parent_event_ids)
        self._state = replace(self._state, events=events, causality=causal)
        return event

    def step(self, dt: float | None = None, updater: Callable[[WorldState, float], WorldState] | None = None) -> WorldState:
        h = self._state.clock.dt if dt is None else float(dt)
        if not math.isfinite(h) or h <= 0:
            raise ValueError("dt must be finite and > 0")
        current = self._state
        candidate = current if updater is None else updater(current, h)
        if not isinstance(candidate, WorldState):
            raise TypeError("world updater must return WorldState")
        # U0 owns the global clock: a physics/biology layer cannot silently alter it.
        candidate = replace(candidate, clock=current.clock.advance(h))
        if candidate.identity.world_id != current.identity.world_id:
            raise ValueError("world-step updater cannot silently replace world identity")
        if candidate.events.events and candidate.events.events[-1].time > candidate.clock.time + _EPS:
            raise ValueError("event time cannot exceed world time")
        # Re-run the state invariants after every world transition.
        candidate = WorldState(identity=candidate.identity, space=candidate.space, clock=candidate.clock,
                               objects=candidate.objects, laws=candidate.laws, events=candidate.events,
                               causality=candidate.causality, metadata=candidate.metadata)
        self._state = candidate
        return candidate

    def snapshot(self) -> WorldState:
        return self._state
