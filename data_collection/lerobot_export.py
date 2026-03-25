#!/usr/bin/env python3
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message


DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def flatten_dict(d, parent_key="", sep="."):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif isinstance(d, (list, tuple, np.ndarray)):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}"
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)


def detect_rosbag_format(bag_path: str) -> Optional[str]:
    if os.path.isdir(bag_path):
        mcap_files = [f for f in os.listdir(bag_path) if f.endswith(".mcap")]
        if mcap_files:
            return "mcap"
        db3_files = [f for f in os.listdir(bag_path) if f.endswith(".db3")]
        if db3_files:
            return "sqlite3"

    reader = rosbag2_py.SequentialReader()
    for storage_id in ["mcap", "sqlite3"]:
        try:
            reader.open(
                rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                rosbag2_py.ConverterOptions("", ""),
            )
            return storage_id
        except Exception:
            continue
    return None


def load_lerobot_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        config = json.load(f)

    if "features" not in config or not isinstance(config["features"], list) or not config["features"]:
        raise ValueError("LeRobot config must include a non-empty 'features' list")

    for spec in config["features"]:
        if "name" not in spec or "topic" not in spec:
            raise ValueError("Each feature spec must include 'name' and 'topic'")
        if "fields" not in spec or not isinstance(spec["fields"], list) or not spec["fields"]:
            raise ValueError(f"Feature '{spec.get('name', '<unnamed>')}' must include a non-empty 'fields' list")
        spec.setdefault("dtype", "float32")
        spec.setdefault("shape", [len(spec["fields"])])
        if isinstance(spec["shape"], tuple):
            spec["shape"] = list(spec["shape"])

    config.setdefault("fps", 30)
    config.setdefault("reference_topic", config["features"][0]["topic"])
    config.setdefault("topic_latency_ns", {})
    config.setdefault("default_task", "task")
    config.setdefault("robot_type", None)
    config.setdefault("codebase_version", "v3.0")
    config.setdefault("include_timestamp_ns", True)
    return config


def _sanitize_topic_to_feature_name(topic: str) -> str:
    name = topic.strip("/").replace("/", ".")
    name = "".join(ch if (ch.isalnum() or ch in "._") else "_" for ch in name)
    while ".." in name:
        name = name.replace("..", ".")
    if not name:
        name = "topic"
    return f"ros.{name}"


def _is_numeric_scalar(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool))


def _estimate_fps(timestamps_ns: List[int]) -> int:
    if len(timestamps_ns) < 2:
        return 30
    dts = []
    prev = timestamps_ns[0]
    for t in timestamps_ns[1:]:
        dt = t - prev
        if dt > 0:
            dts.append(dt)
        prev = t
    if not dts:
        return 30
    median_dt = sorted(dts)[len(dts) // 2]
    if median_dt <= 0:
        return 30
    fps = int(round(1e9 / median_dt))
    return max(1, min(240, fps))


def infer_lerobot_config_from_bag(
    bag_path: str,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    max_fields_per_topic: int = 64,
) -> Dict:
    storage_id = detect_rosbag_format(bag_path)
    if not storage_id:
        raise RuntimeError(f"Unable to detect rosbag format for {bag_path}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    sample_payload = {}
    topic_counts = {}
    topic_timestamps = {}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if start_ns is not None and t < start_ns:
            continue
        if end_ns is not None and t > end_ns:
            continue
        if topic not in topic_types:
            continue

        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_timestamps.setdefault(topic, []).append(int(t))

        if topic in sample_payload:
            continue

        try:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            flat = flatten_dict(message_to_ordereddict(msg))
        except Exception:
            continue

        numeric_fields = [k for k, v in flat.items() if _is_numeric_scalar(v)]
        if numeric_fields:
            sample_payload[topic] = sorted(numeric_fields)[:max_fields_per_topic]

    candidate_topics = [t for t in topic_counts.keys() if t in sample_payload]
    if not candidate_topics:
        raise RuntimeError("No numeric ROS topics found in rosbag for auto LeRobot export")

    reference_topic = max(candidate_topics, key=lambda t: topic_counts.get(t, 0))
    fps = _estimate_fps(topic_timestamps.get(reference_topic, []))

    features = []
    for topic in sorted(candidate_topics):
        fields = sample_payload[topic]
        feature_name = _sanitize_topic_to_feature_name(topic)
        features.append(
            {
                "name": feature_name,
                "topic": topic,
                "fields": fields,
                "dtype": "float32",
                "shape": [len(fields)],
                "names": {"axes": fields},
            }
        )

    return {
        "fps": fps,
        "reference_topic": reference_topic,
        "topic_latency_ns": {},
        "default_task": "task",
        "robot_type": None,
        "codebase_version": "v3.0",
        "include_timestamp_ns": True,
        "features": features,
    }


def _default_value(dtype: str):
    if dtype in ("float16", "float32", "float64"):
        return float("nan")
    if dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        return 0
    if dtype == "bool":
        return False
    return None


def _cast_scalar(value, dtype: str):
    if value is None:
        return _default_value(dtype)
    try:
        if dtype in ("float16", "float32", "float64"):
            return float(value)
        if dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
            return int(value)
        if dtype == "bool":
            return bool(value)
        if dtype == "string":
            return str(value)
    except Exception:
        return _default_value(dtype)
    return value


def _sample_feature_values(
    entries: List[Tuple[int, Dict]],
    ref_timestamps_ns: List[int],
    fields: List[str],
    dtype: str,
    shape: List[int],
) -> List:
    if not ref_timestamps_ns:
        return []

    if not entries:
        fill = _default_value(dtype)
        if len(shape) == 1 and shape[0] == 1:
            return [fill] * len(ref_timestamps_ns)
        return [[fill for _ in fields] for _ in ref_timestamps_ns]

    ts = [t for t, _ in entries]
    payloads = [p for _, p in entries]

    out = []
    idx = 0
    n = len(ts)

    for ref_t in ref_timestamps_ns:
        while idx + 1 < n and ts[idx + 1] <= ref_t:
            idx += 1

        source = payloads[idx] if ts[idx] <= ref_t else payloads[0]
        values = [_cast_scalar(source.get(field), dtype) for field in fields]

        if len(shape) == 1 and shape[0] == 1:
            out.append(values[0])
        else:
            out.append(values)

    return out


def _build_reference_timestamps(
    topic_messages: Dict[str, List[Tuple[int, Dict]]],
    reference_topic: Optional[str],
    fps: int,
    start_ns: Optional[int],
    end_ns: Optional[int],
) -> List[int]:
    if reference_topic and reference_topic in topic_messages and topic_messages[reference_topic]:
        return [t for t, _ in topic_messages[reference_topic]]

    mins = []
    maxs = []
    for entries in topic_messages.values():
        if entries:
            mins.append(entries[0][0])
            maxs.append(entries[-1][0])

    if not mins:
        return []

    t_start = max(mins) if start_ns is None else max(start_ns, min(mins))
    t_end = min(maxs) if end_ns is None else min(end_ns, max(maxs))
    if t_end < t_start:
        return []

    dt = max(1, int(1e9 / max(1, fps)))
    return list(range(t_start, t_end + 1, dt))


class LeRobotPhase1Writer:
    def __init__(self, dataset_root: str, config: Dict):
        self.dataset_root = dataset_root
        self.config = config
        self.rows = []
        self.episodes = []
        self.task_to_index = {}
        self.tasks = []

        self.features = {spec["name"]: {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "names": spec.get("names")
        } for spec in self.config["features"]}

        if self.config.get("include_timestamp_ns", True):
            self.features["timestamp_ns"] = {"dtype": "int64", "shape": [1], "names": None}

        self.features = {**self.features, **DEFAULT_FEATURES}

    def _task_index(self, task_name: str) -> int:
        if task_name not in self.task_to_index:
            self.task_to_index[task_name] = len(self.task_to_index)
            self.tasks.append(task_name)
        return self.task_to_index[task_name]

    def add_episode(self, stage_name: str, task_name: str, ref_timestamps_ns: List[int], sampled: Dict[str, List]):
        if not ref_timestamps_ns:
            return

        episode_index = len(self.episodes)
        task_index = self._task_index(task_name)
        dataset_from = len(self.rows)
        t0 = ref_timestamps_ns[0]

        for frame_idx, ts_ns in enumerate(ref_timestamps_ns):
            row = {
                "index": dataset_from + frame_idx,
                "episode_index": episode_index,
                "frame_index": frame_idx,
                "timestamp": float(ts_ns - t0) / 1e9,
                "task_index": task_index,
            }
            if self.config.get("include_timestamp_ns", True):
                row["timestamp_ns"] = int(ts_ns)

            for spec in self.config["features"]:
                name = spec["name"]
                row[name] = sampled[name][frame_idx]

            self.rows.append(row)

        length = len(ref_timestamps_ns)
        self.episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": length,
                "stage_name": stage_name,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": dataset_from,
                "dataset_to_index": dataset_from + length,
            }
        )

    def has_episodes(self) -> bool:
        return len(self.episodes) > 0

    def finalize(self):
        if not self.rows:
            return

        try:
            pd = __import__("pandas")
        except ImportError as e:
            raise RuntimeError(
                "LeRobot export requires pandas and pyarrow. Please install python3-pandas and python3-pyarrow."
            ) from e

        data_dir = os.path.join(self.dataset_root, "data", "chunk-000")
        episodes_dir = os.path.join(self.dataset_root, "meta", "episodes", "chunk-000")
        meta_dir = os.path.join(self.dataset_root, "meta")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(episodes_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        data_df = pd.DataFrame(self.rows)
        data_df.to_parquet(os.path.join(data_dir, "file-000.parquet"), index=False)

        episodes_df = pd.DataFrame(self.episodes)
        episodes_df.to_parquet(os.path.join(episodes_dir, "file-000.parquet"), index=False)

        tasks_df = pd.DataFrame({"task_index": [self.task_to_index[t] for t in self.tasks]}, index=pd.Index(self.tasks, name="task"))
        tasks_df.to_parquet(os.path.join(meta_dir, "tasks.parquet"))

        info = {
            "codebase_version": self.config.get("codebase_version", "v3.0"),
            "robot_type": self.config.get("robot_type"),
            "total_episodes": len(self.episodes),
            "total_frames": len(self.rows),
            "total_tasks": len(self.tasks),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": int(self.config.get("fps", 30)),
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "features": self.features,
        }

        with open(os.path.join(meta_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=4)


def export_rosbag_episode(
    bag_path: str,
    writer: LeRobotPhase1Writer,
    stage_name: str,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
):
    config = writer.config
    feature_specs = config["features"]
    reference_topic = config.get("reference_topic")
    fps = int(config.get("fps", 30))
    topic_latency = config.get("topic_latency_ns", {})

    needed_topics = {spec["topic"] for spec in feature_specs}
    if reference_topic:
        needed_topics.add(reference_topic)

    storage_id = detect_rosbag_format(bag_path)
    if not storage_id:
        print(f"Error: Unable to detect rosbag format for {bag_path}")
        return False

    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
            rosbag2_py.ConverterOptions("", ""),
        )
    except Exception as e:
        print(f"Error opening rosbag for LeRobot export: {e}")
        return False

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    topic_messages: Dict[str, List[Tuple[int, Dict]]] = {topic: [] for topic in needed_topics}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in needed_topics:
            continue
        if start_ns is not None and t < start_ns:
            continue
        if end_ns is not None and t > end_ns:
            continue

        if topic not in topic_types:
            continue

        try:
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            flat = flatten_dict(message_to_ordereddict(msg))
        except Exception:
            continue

        adjusted_t = int(t) - int(topic_latency.get(topic, 0))
        topic_messages[topic].append((adjusted_t, flat))

    for topic in topic_messages:
        topic_messages[topic].sort(key=lambda x: x[0])

    ref_timestamps_ns = _build_reference_timestamps(topic_messages, reference_topic, fps, start_ns, end_ns)
    if not ref_timestamps_ns:
        print(f"Warning: no reference timestamps for stage '{stage_name}', skipping LeRobot export")
        return False

    sampled = {}
    for spec in feature_specs:
        name = spec["name"]
        topic = spec["topic"]
        fields = spec["fields"]
        dtype = spec.get("dtype", "float32")
        shape = spec.get("shape", [len(fields)])
        sampled[name] = _sample_feature_values(topic_messages.get(topic, []), ref_timestamps_ns, fields, dtype, shape)

    task_map = config.get("task_by_stage", {})
    task_name = task_map.get(stage_name, config.get("default_task", "task"))
    writer.add_episode(stage_name, task_name, ref_timestamps_ns, sampled)
    return True
