"""AIRE UP — persistence, branching and deterministic replay.

UP adds an auditable persistence layer above the immutable U0 WorldState.  It
never mutates a historical checkpoint: branches are new world identities with
an explicit parent, and replay uses a serializable command log rather than
serializing executable callables.
"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict, replace
from typing import Any, Mapping
import copy
import json
import hashlib
from pathlib import Path
import numpy as np

from .core import (
    WorldCore, WorldEvent, WorldIdentity, WorldObject, WorldState,
    Space3D, SimulationClock, LawRegistry, EventLog, CausalLedger,
)

UP_SCHEMA = "UP.1"


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "dtype": str(value.dtype), "value": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {"__type__": type(value).__name__, "value": {k: _encode(v) for k, v in asdict(value).items()}}
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # U0 permits arbitrary state payloads, so persistence must fail closed
    # rather than silently producing a non-replayable representation.
    raise TypeError(f"non-persistable value: {type(value).__name__}")


def _decode_array(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__type__") == "ndarray":
        return np.asarray(value["value"], dtype=value.get("dtype", "float64"))
    if isinstance(value, list):
        return [_decode_array(v) for v in value]
    if isinstance(value, dict):
        return {k: _decode_array(v) for k, v in value.items()}
    return value


def _object_to_dict(obj: WorldObject) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "kind": obj.kind,
        "position": _encode(obj.position),
        "velocity": _encode(obj.velocity),
        "state": _encode(dict(obj.state)),
        "tags": list(obj.tags),
    }


def _object_from_dict(data: Mapping[str, Any]) -> WorldObject:
    return WorldObject(
        object_id=str(data["object_id"]), kind=str(data["kind"]),
        position=_decode_array(data["position"]), velocity=_decode_array(data["velocity"]),
        state=_decode_array(data.get("state", {})), tags=tuple(data.get("tags", ())),
    )


def serialize_world(state: WorldState) -> dict[str, Any]:
    """Return a canonical, JSON-safe snapshot of a WorldState."""
    return {
        "schema": UP_SCHEMA,
        "identity": {
            "world_id": state.identity.world_id,
            "parent_world_id": state.identity.parent_world_id,
            "seed": state.identity.seed,
            "schema_version": state.identity.schema_version,
        },
        "space": {"origin": _encode(state.space.origin),
                  "bounds_min": _encode(state.space.bounds_min) if state.space.bounds_min is not None else None,
                  "bounds_max": _encode(state.space.bounds_max) if state.space.bounds_max is not None else None},
        "clock": {"time": state.clock.time, "step_index": state.clock.step_index, "dt": state.clock.dt},
        "objects": [_object_to_dict(o) for o in sorted(state.objects.values(), key=lambda x: x.object_id)],
        "laws": {k: _encode(dict(v)) for k, v in sorted(state.laws.laws.items())},
        "events": [{"event_id": e.event_id, "time": e.time, "event_type": e.event_type,
                    "source_ids": list(e.source_ids), "target_ids": list(e.target_ids),
                    "parent_event_ids": list(e.parent_event_ids), "payload": _encode(dict(e.payload))}
                   for e in state.events.events],
        "causality": {k: list(v) for k, v in sorted(state.causality.parents.items())},
        "metadata": _encode(dict(state.metadata)),
    }


def deserialize_world(data: Mapping[str, Any]) -> WorldState:
    if data.get("schema") != UP_SCHEMA:
        raise ValueError(f"unsupported world snapshot schema: {data.get('schema')!r}")
    ident = data["identity"]
    space = data["space"]
    clock = data["clock"]
    identity = WorldIdentity(**ident)
    world_space = Space3D(origin=_decode_array(space["origin"]),
                          bounds_min=_decode_array(space["bounds_min"]) if space["bounds_min"] is not None else None,
                          bounds_max=_decode_array(space["bounds_max"]) if space["bounds_max"] is not None else None)
    objects = {}
    for raw_obj in data.get("objects", []):
        obj = _object_from_dict(raw_obj)
        objects[obj.object_id] = obj
    events = tuple(WorldEvent(event_id=e["event_id"], time=e["time"], event_type=e["event_type"],
                              source_ids=tuple(e.get("source_ids", ())), target_ids=tuple(e.get("target_ids", ())),
                              parent_event_ids=tuple(e.get("parent_event_ids", ())),
                              payload=_decode_array(e.get("payload", {}))) for e in data.get("events", []))
    return WorldState(
        identity=identity, space=world_space,
        clock=SimulationClock(**clock), objects=objects,
        laws=LawRegistry(_decode_array(data.get("laws", {}))),
        events=EventLog(events),
        causality=CausalLedger({k: tuple(v) for k, v in data.get("causality", {}).items()}),
        metadata=_decode_array(data.get("metadata", {})),
    )


def canonical_world_json(state: WorldState) -> str:
    """Canonical JSON representation used for cross-run reproducibility."""
    return json.dumps(serialize_world(state), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compare_worlds(a: WorldState, b: WorldState) -> dict[str, Any]:
    """Return a deterministic, component-level first divergence report."""
    if a.state_digest() == b.state_digest():
        return {"equal": True, "first_divergence": None}
    sa, sb = serialize_world(a), serialize_world(b)
    for component in ("identity", "space", "clock", "objects", "laws", "events", "causality", "metadata"):
        if sa.get(component) != sb.get(component):
            return {
                "equal": False,
                "first_divergence": component,
                "left_digest": a.state_digest(),
                "right_digest": b.state_digest(),
            }
    return {"equal": False, "first_divergence": "digest_only", "left_digest": a.state_digest(), "right_digest": b.state_digest()}


def serialize_commands(commands: tuple["WorldCommand", ...] | list["WorldCommand"]) -> list[dict[str, Any]]:
    """Serialize a command log without executable code or callbacks."""
    return [{"sequence": int(c.sequence), "operation": c.operation, "payload": _encode(dict(c.payload))} for c in commands]


def deserialize_commands(data: list[Mapping[str, Any]]) -> tuple["WorldCommand", ...]:
    commands = tuple(WorldCommand(int(c["sequence"]), str(c["operation"]), _decode_array(c.get("payload", {}))) for c in data)
    expected = list(range(len(commands)))
    actual = [c.sequence for c in commands]
    if actual != expected:
        raise ValueError("command sequence is not contiguous")
    return commands


@dataclass(frozen=True)
class WorldCheckpoint:
    checkpoint_id: str
    world_digest: str
    state: Mapping[str, Any]
    schema: str = UP_SCHEMA

    @classmethod
    def capture(cls, world: WorldState, checkpoint_id: str) -> "WorldCheckpoint":
        payload = serialize_world(world)
        # The checkpoint integrity value is the canonical U0 world digest.
        # This makes restore verify semantic world state, not merely the JSON
        # container representation.
        digest = world.state_digest()
        return cls(str(checkpoint_id), digest, payload)

    def restore(self) -> WorldState:
        world = deserialize_world(copy.deepcopy(dict(self.state)))
        if world.state_digest() != self.world_digest:
            raise ValueError("checkpoint integrity digest mismatch")
        return world

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"checkpoint_id": self.checkpoint_id,
                                          "world_digest": self.world_digest,
                                          "schema": self.schema,
                                          "state": self.state}, sort_keys=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WorldCheckpoint":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["checkpoint_id"], data["world_digest"], data["state"], data.get("schema", UP_SCHEMA))


@dataclass(frozen=True)
class WorldCommand:
    sequence: int
    operation: str
    payload: Mapping[str, Any]


class WorldSession:
    """Command-recording facade for deterministic world experiments."""
    def __init__(self, initial: WorldState):
        self.core = WorldCore(initial)
        self.initial_snapshot = serialize_world(initial)
        self.commands: list[WorldCommand] = []

    @property
    def state(self) -> WorldState:
        return self.core.state

    def _record(self, operation: str, payload: Mapping[str, Any]) -> None:
        self.commands.append(WorldCommand(len(self.commands), operation, copy.deepcopy(dict(payload))))

    def add_object(self, obj: WorldObject) -> WorldState:
        self.core.add_object(obj)
        self._record("add_object", {"object": _object_to_dict(obj)})
        return self.state

    def update_object(self, obj: WorldObject) -> WorldState:
        self.core._state = self.core.state.update_object(obj)
        self._record("update_object", {"object": _object_to_dict(obj)})
        return self.state

    def remove_object(self, object_id: str) -> WorldState:
        self.core._state = self.core.state.remove_object(object_id)
        self._record("remove_object", {"object_id": str(object_id)})
        return self.state

    def emit(self, event_type: str, *, source_ids=(), target_ids=(), parent_event_ids=(), payload=None) -> WorldEvent:
        event = self.core.emit(event_type, source_ids=source_ids, target_ids=target_ids,
                               parent_event_ids=parent_event_ids, payload=payload)
        self._record("emit", {"event_id": event.event_id, "time": event.time, "event_type": event_type,
                               "source_ids": list(source_ids), "target_ids": list(target_ids),
                               "parent_event_ids": list(parent_event_ids), "payload": _encode(payload or {})})
        return event

    def step(self, dt: float | None = None) -> WorldState:
        h = self.state.clock.dt if dt is None else float(dt)
        self.core.step(h)
        self._record("step", {"dt": h})
        return self.state

    def checkpoint(self, checkpoint_id: str) -> WorldCheckpoint:
        return WorldCheckpoint.capture(self.state, checkpoint_id)

    def branch(self, branch_id: str, *, seed: int | None = None) -> "WorldSession":
        parent = self.state.identity.world_id
        identity = WorldIdentity(world_id=str(branch_id), parent_world_id=parent,
                                 seed=self.state.identity.seed if seed is None else int(seed),
                                 schema_version=self.state.identity.schema_version)
        child = deserialize_world(serialize_world(self.state))
        child = replace(child, identity=identity)
        return WorldSession(child)

    def replay(self) -> WorldState:
        return self.replay_from_commands(self.commands)

    def command_log_payload(self) -> list[dict[str, Any]]:
        return serialize_commands(self.commands)

    def save_command_log(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "schema": UP_SCHEMA,
            "initial_snapshot": self.initial_snapshot,
            "commands": self.command_log_payload(),
        }, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    @staticmethod
    def load_command_log(path: str | Path) -> tuple[dict[str, Any], tuple[WorldCommand, ...]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != UP_SCHEMA:
            raise ValueError("unsupported command-log schema")
        return data["initial_snapshot"], deserialize_commands(data.get("commands", []))

    def replay_from_commands(self, commands: tuple[WorldCommand, ...] | list[WorldCommand]) -> WorldState:
        replay = WorldSession(deserialize_world(copy.deepcopy(self.initial_snapshot)))
        for cmd in deserialize_commands(serialize_commands(commands)):
            p = cmd.payload
            if cmd.operation == "add_object": replay.add_object(_object_from_dict(p["object"]))
            elif cmd.operation == "update_object": replay.update_object(_object_from_dict(p["object"]))
            elif cmd.operation == "remove_object": replay.remove_object(p["object_id"])
            elif cmd.operation == "emit":
                event = WorldEvent(event_id=p["event_id"], time=float(p["time"]), event_type=p["event_type"],
                                   source_ids=tuple(p["source_ids"]), target_ids=tuple(p["target_ids"]),
                                   parent_event_ids=tuple(p["parent_event_ids"]), payload=_decode_array(p["payload"]))
                replay.core._state = replace(replay.state,
                    events=replay.state.events.append(event),
                    causality=replay.state.causality.add(event.event_id, event.parent_event_ids))
            elif cmd.operation == "step": replay.step(p["dt"])
            else: raise ValueError(f"unknown replay operation: {cmd.operation}")
        return replay.state

    def branch_comparison(self, other: "WorldSession") -> dict[str, Any]:
        return compare_worlds(self.state, other.state)

    def command_log(self) -> tuple[WorldCommand, ...]:
        return tuple(self.commands)
