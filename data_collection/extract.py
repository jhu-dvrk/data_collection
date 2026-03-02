#!/usr/bin/env python3
import cv2
import json
import os
import argparse
import sys
import threading
import csv
import rosbag2_py
import multiprocessing
import math
import numpy as np
import re
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
from .common import parse_stage_timestamp

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    if isinstance(d, (dict,)) or (hasattr(d, 'items') and callable(getattr(d, 'items'))):
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

def detect_rosbag_format(bag_path):
    if os.path.isdir(bag_path):
        mcap_files = [f for f in os.listdir(bag_path) if f.endswith('.mcap')]
        if mcap_files: return 'mcap'
        db3_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
        if db3_files: return 'sqlite3'
    
    reader = rosbag2_py.SequentialReader()
    for storage_id in ['mcap', 'sqlite3']:
        try:
            reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                        rosbag2_py.ConverterOptions('', ''))
            return storage_id
        except Exception: continue
    return None

def rosbag_to_csv(bag_path, output_dir, start_ns=None, end_ns=None):
    print(f"Converting ROS bag {bag_path} to CSV...")
    storage_id = detect_rosbag_format(bag_path)
    if not storage_id:
        print(f"Error: Unable to detect rosbag format for {bag_path}")
        return
    
    # Pass 1: Discover all fields and vector sizes
    print("Pass 1: Discovering columns and vector sizes...")
    topic_fields = {} # topic -> set of field names
    reader_p1 = rosbag2_py.SequentialReader()
    try:
        reader_p1.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                       rosbag2_py.ConverterOptions('', ''))
    except Exception as e:
        print(f"Error opening rosbag for Pass 1: {e}")
        return

    topic_types = {topic.name: topic.type for topic in reader_p1.get_all_topics_and_types()}
    
    while reader_p1.has_next():
        (topic, data, t) = reader_p1.read_next()
        if start_ns is not None and t < start_ns: continue
        if end_ns is not None and t > end_ns: continue
        
        if topic not in topic_fields:
            topic_fields[topic] = set(['timestamp'])
            
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        msg_dict = flatten_dict(message_to_ordereddict(msg))
        topic_fields[topic].update(msg_dict.keys())
    
    del reader_p1 # Close reader for re-opening

    # Pass 2: Write data with unified header
    print("Pass 2: Writing CSV data...")
    reader_p2 = rosbag2_py.SequentialReader()
    try:
        reader_p2.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                       rosbag2_py.ConverterOptions('', ''))
    except Exception as e:
        print(f"Error opening rosbag for Pass 2: {e}")
        return
    
    writers = {}
    files = {}

    try:
        while reader_p2.has_next():
            (topic, data, t) = reader_p2.read_next()
            if start_ns is not None and t < start_ns: continue
            if end_ns is not None and t > end_ns: continue
            
            if topic not in writers:
                csv_name = topic.replace('/', '_').strip('_') + ".csv"
                csv_path = os.path.join(output_dir, csv_name)
                f = open(csv_path, 'w', newline='')
                files[topic] = f
                
                f.write(f"# Topic: {topic}\n# Type: {topic_types[topic]}\n")
                
                # Natural sort field names
                fieldnames = sorted(list(topic_fields[topic]), key=natural_sort_key)
                if 'timestamp' in fieldnames:
                    fieldnames.remove('timestamp')
                    fieldnames = ['timestamp'] + fieldnames
                
                writers[topic] = csv.DictWriter(f, fieldnames=fieldnames)
                writers[topic].writeheader()
            
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            row = {'timestamp': t}
            row.update(flatten_dict(message_to_ordereddict(msg)))
            writers[topic].writerow(row)
    except Exception as e:
        print(f"Error reading bag: {e}")
    finally:
        for f in files.values(): f.close()
    print("Finished CSV conversion.")

def process_video_chunk(args):
    video_path, output_dir, timestamps, start_frame_idx, output_format, video_basename, is_ns, is_ms = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    saved_count = 0
    for i, ts in enumerate(timestamps):
        ret, frame = cap.read()
        if not ret: break
        
        if is_ns: ts_str = f"{int(ts) / 1e9:.9f}"
        elif is_ms: ts_str = f"{float(ts) / 1000.0:.6f}"
        else: ts_str = f"{float(ts):.6f}"
        
        image_name = f"{video_basename}_{ts_str}.{output_format}"
        cv2.imwrite(os.path.join(output_dir, image_name), frame)
        saved_count += 1
    cap.release()
    return saved_count

def extract_video_range(video_path, output_path, start_frame, end_frame, fps, original_indices=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if original_indices is not None:
        for idx in original_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            out.write(frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        
    cap.release()
    out.release()
    return True

def extract_session_data(json_path, output_dir, formats, num_jobs, start_acq=None, end_acq=None, latency_s=0.0):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    # Check for dc_sidecar_v1 type
    if data.get("type") != "dc_sidecar_v1":
        print(f"Warning: JSON file {json_path} does not have type 'dc_sidecar_v1'. "
              f"Found: {data.get('type')}. Proceeding anyway.")

    video_filename = data.get("video_file")
    video_path = None
    if video_filename:
        video_path = os.path.join(os.path.dirname(json_path), video_filename)
    if not video_path or not os.path.exists(video_path):
        base = os.path.splitext(json_path)[0]
        if os.path.exists(base + ".mp4"):
            video_path = base + ".mp4"

    if not video_path: return

    # acquisition_time = cpu_ts - latency
    latency_s = data.get("estimated_latency", latency_s)
    latency_ns = int(latency_s * 1e9)

    timestamps = data.get("timestamps_ns", data.get("timestamps", data.get("timestamps_ms")))
    is_ns = "timestamps_ns" in data
    is_ms = "timestamps_ms" in data and not is_ns

    if not timestamps and "frames" in data:
        frames = data["frames"]
        if frames and isinstance(frames[0], dict):
            timestamps = [f.get("cpu_ts", f.get("gst_ts", 0)) for f in frames]
            is_ns = True; is_ms = False

    if not timestamps: return

    indices = []
    for i, ts in enumerate(timestamps):
        acq_ts = ts - latency_ns
        if (start_acq is None or acq_ts >= start_acq) and (end_acq is None or acq_ts <= end_acq):
            indices.append(i)
    
    if not indices: return
    
    start_idx = indices[0]
    end_idx = indices[-1]
    filtered_timestamps = [timestamps[i] for i in indices]
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    for fmt in formats:
        if fmt == 'mp4':
            fps = data.get("fps", 30.0)
            out_name = f"{video_basename}.mp4"
            print(f"Extracting video segment: {out_name}")
            extract_video_range(video_path, os.path.join(output_dir, out_name), start_idx, end_idx, fps, indices)
        else:
            tasks = []
            chunk_size = math.ceil(len(indices) / num_jobs)
            for i in range(num_jobs):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, len(indices))
                if s >= len(indices): break
                tasks.append((video_path, output_dir, [filtered_timestamps[k] for k in range(s, e)], indices[s], fmt, video_basename, is_ns, is_ms))
            
            if num_jobs > 1:
                with multiprocessing.Pool(processes=num_jobs) as pool: pool.map(process_video_chunk, tasks)
            else:
                process_video_chunk(tasks[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Session directory", required=True)
    parser.add_argument("-t", "--tags", help="Tags JSON file")
    parser.add_argument("-a", "--all", action="store_true", help="Extract all data")
    parser.add_argument("-f", "--format", choices=['jpg', 'png', 'mp4'], action='append', help="Output format(s).")
    parser.add_argument("-j", "--jobs", type=int)
    args = parser.parse_args()

    if not args.tags and not args.all:
        print("Error: Either -t (tags) or -a (all) must be specified.")
        sys.exit(1)

    formats = args.format if args.format else ['mp4']
    num_jobs = args.jobs or max(1, (os.cpu_count() or 1) // 2)
    
    if not os.path.exists(args.directory):
        print(f"CRITICAL: Session directory does not exist: {args.directory}")
        sys.exit(1)

    index_path = os.path.join(args.directory, "index.json")
    if not os.path.exists(index_path):
        print(f"CRITICAL: index.json not found in {args.directory}")
        sys.exit(1)

    with open(index_path, 'r') as f: index_data = json.load(f)
    
    videos = index_data.get("videos", [])
    rosbag_name = index_data.get("rosbags", index_data.get("rosbag", {}).get("name") if isinstance(index_data.get("rosbag"), dict) else None)

    base_extracted_dir = os.path.join(args.directory, "extracted")
    os.makedirs(base_extracted_dir, exist_ok=True)

    extraction_targets = []
    if args.all:
        extraction_targets.append({"name": "full_session", "start": None, "end": None})
    if args.tags:
        tags_path = args.tags if os.path.isabs(args.tags) else os.path.join(args.directory, args.tags)
        with open(tags_path, 'r') as f: tags_data = json.load(f)
        
        source_v_name = tags_data.get("video_file")
        source_v_latency = 0.0
        source_v_ts = []
        if source_v_name:
            for v in videos:
                if v["file"] == source_v_name:
                    v_base = os.path.splitext(v["file"])[0]
                    v_json = os.path.join(args.directory, f"{v_base}.json")
                    if os.path.exists(v_json):
                        with open(v_json, 'r') as vf:
                            v_data = json.load(vf)
                            source_v_latency = v_data.get("estimated_latency", 0.0)
                            source_v_ts = v_data.get("timestamps_ns", v_data.get("timestamps", []))
                    break

        stage_counts = {}
        for stage in tags_data.get("stages", []):
            name = stage["name"]
            
            ts_s, f_s = parse_stage_timestamp(stage.get("start"))
            ts_e, f_e = parse_stage_timestamp(stage.get("end"))
            
            def resolve_acq(ts, f):
                if f is not None and source_v_ts:
                    if 0 <= f < len(source_v_ts):
                        return source_v_ts[f] - int(source_v_latency * 1e9)
                if ts:
                    return ts - int(source_v_latency * 1e9)
                return None

            start_val = resolve_acq(ts_s, f_s)
            end_val = resolve_acq(ts_e, f_e)
                
            stage_counts[name] = stage_counts.get(name, 0) + 1
            dir_name = f"{name}_{stage_counts[name]:03d}"
            extraction_targets.append({"name": dir_name, "start": start_val, "end": end_val})

    for target in extraction_targets:
        target_dir = os.path.join(base_extracted_dir, target["name"])
        os.makedirs(target_dir, exist_ok=True)
        print(f"Processing target: {target['name']}")
        
        for v in videos:
            v_base = os.path.splitext(v["file"])[0]
            v_json = os.path.join(args.directory, f"{v_base}.json")
            if os.path.exists(v_json):
                extract_session_data(v_json, target_dir, formats, num_jobs, target["start"], target["end"])
        
        if rosbag_name:
            bag_path = os.path.join(args.directory, rosbag_name)
            if os.path.exists(bag_path):
                rosbag_to_csv(bag_path, target_dir, target["start"], target["end"])

if __name__ == "__main__":
    main()
