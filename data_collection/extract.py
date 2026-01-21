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
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or (hasattr(v, 'items') and callable(getattr(v, 'items'))):
            items.extend(flatten_dict(dict(v), new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def detect_rosbag_format(bag_path):
    """Detect the storage format of a rosbag2 directory.
    Returns 'mcap', 'sqlite3', or None if unable to detect.
    """
    # Check for common storage files
    if os.path.isdir(bag_path):
        # Look for mcap file
        mcap_files = [f for f in os.listdir(bag_path) if f.endswith('.mcap')]
        if mcap_files:
            return 'mcap'
        
        # Look for sqlite3 db file
        db3_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
        if db3_files:
            return 'sqlite3'
    
    # Try opening with each storage plugin
    reader = rosbag2_py.SequentialReader()
    for storage_id in ['mcap', 'sqlite3']:
        try:
            reader.open(
                rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                rosbag2_py.ConverterOptions('', '')
            )
            return storage_id
        except Exception:
            continue
    
    return None

def rosbag_to_csv(bag_path, output_dir):
    print(f"Converting ROS bag {bag_path} to CSV...")
    
    # Detect storage format
    storage_id = detect_rosbag_format(bag_path)
    if not storage_id:
        print(f"Error: Unable to detect rosbag format for {bag_path}")
        return
    
    print(f"Detected rosbag storage format: {storage_id}")
    
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
            rosbag2_py.ConverterOptions('', '')
        )
    except Exception as e:
        print(f"Error opening rosbag with {storage_id} format: {e}")
        return

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    writers = {}
    files = {}

    try:
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            
            msg_dict = flatten_dict(message_to_ordereddict(msg))
            
            if topic not in writers:
                csv_name = topic.replace('/', '_').strip('_') + ".csv"
                csv_path = os.path.join(output_dir, csv_name)
                f = open(csv_path, 'w', newline='')
                files[topic] = f
                
                # Write metadata header
                f.write(f"# Topic: {topic}\n")
                f.write(f"# Type: {topic_types[topic]}\n")
                
                fieldnames = ['timestamp'] + list(msg_dict.keys())
                writers[topic] = csv.DictWriter(f, fieldnames=fieldnames)
                writers[topic].writeheader()
                
            row = {'timestamp': t}
            row.update(msg_dict)
            writers[topic].writerow(row)
    except Exception as e:
        print(f"Error reading bag: {e}")
    finally:
        for f in files.values():
            f.close()
    print(f"Finished CSV conversion.")

def process_video_chunk(args):
    """
    Worker function to process a chunk of the video.
    args: (video_path, output_dir, timestamps_chunk, start_frame_idx, image_format, video_basename, is_ns, is_ms)
    """
    video_path, output_dir, timestamps, start_frame_idx, image_format, video_basename, is_ns, is_ms = args
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path} in worker.")
        return 0

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    
    saved_count = 0
    for i, ts in enumerate(timestamps):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Format timestamp in filename
        if is_ns:
            ts_str = f"{int(ts) / 1e9:.9f}"
        elif is_ms:
            ts_str = f"{float(ts) / 1000.0:.6f}"
        else:
            ts_str = f"{float(ts):.6f}"
        
        image_name = f"{video_basename}_{ts_str}.{image_format}"
        image_path = os.path.join(output_dir, image_name)
        
        cv2.imwrite(image_path, frame)
        saved_count += 1
        
    cap.release()
    return saved_count

def extract_frames(input_path, output_dir=None, image_format='png', num_jobs=1):
    if not input_path.lower().endswith('.json'):
        print("Error: Input must be a JSON timestamp file.")
        return

    json_path = input_path
    video_path = None
    
    # Read JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # Determine video path
    video_filename = data.get("video_file")
    if video_filename:
        video_path = os.path.join(os.path.dirname(input_path), video_filename)
        
    if not video_path or not os.path.exists(video_path):
         base = os.path.splitext(input_path)[0]
         possible_vid = base + ".mp4"
         if os.path.exists(possible_vid):
             video_path = possible_vid

    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video file not found. Checked from valid JSON key or implicit name: {video_path}")
        return

    # Load timestamps
    print(f"Loading timestamps from {json_path}...")
    timestamps = data.get("timestamps_ns", data.get("timestamps", data.get("timestamps_ms")))
        
    if not timestamps:
        print("Error: No timestamps found in JSON file")
        return

    is_ns = "timestamps_ns" in data
    is_ms = "timestamps_ms" in data and not is_ns

    # Output directory
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    print(f"[{video_basename}] Processing {video_path}")
    print(f"[{video_basename}] Total timestamps: {len(timestamps)}")
    print(f"[{video_basename}] Output directory: {output_dir}")
    print(f"[{video_basename}] Parallel jobs: {num_jobs}")

    # Prepare chunks
    total_frames = len(timestamps)
    chunk_size = math.ceil(total_frames / num_jobs)
    
    tasks = []
    for i in range(num_jobs):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_frames)
        
        if start_idx >= total_frames:
            break
            
        chunk_timestamps = timestamps[start_idx:end_idx]
        
        # Args: (video_path, output_dir, timestamps_chunk, start_frame_idx, image_format, video_basename, is_ns, is_ms)
        task_args = (
            video_path, 
            output_dir, 
            chunk_timestamps, 
            start_idx, # start_frame_idx assumes 1-to-1 mapping with timestamps idx
            image_format, 
            video_basename, 
            is_ns, 
            is_ms
        )
        tasks.append(task_args)

    # Run in parallel
    total_saved = 0
    if num_jobs > 1:
        with multiprocessing.Pool(processes=num_jobs) as pool:
            results = pool.map(process_video_chunk, tasks)
            total_saved = sum(results)
    else:
        # Sequential fallback (avoid overhead)
        total_saved = process_video_chunk(tasks[0])

    print(f"[{video_basename}] Done. Extracted {total_saved} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from all videos in a session directory using index.json")
    parser.add_argument("-d", "--directory", help="Path to the recording session directory", required=True)
    parser.add_argument("-l", "--list", action="store_true", help="List names of videos in the session but do not extract frames")
    parser.add_argument("-f", "--format", choices=['jpg', 'png'], default='png', help="Output image format (jpg or png, default: png)")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs per video (default: half of available cores)")
    
    args = parser.parse_args()

    # Determine default jobs
    cpu_count = os.cpu_count() or 1
    if args.jobs:
        num_jobs = args.jobs
    else:
        num_jobs = max(1, cpu_count // 2)

    index_path = os.path.join(args.directory, "index.json")
    if not os.path.exists(index_path):
        print(f"Error: index.json not found in {args.directory}")
        sys.exit(1)
        
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
    except Exception as e:
        print(f"Error reading index.json: {e}")
        sys.exit(1)
        
    videos = index_data.get("videos", [])
    if not videos:
        print("No videos found in index.json")
        return

    if args.list:
        session_name = os.path.basename(args.directory.rstrip('/'))
        print(f"Videos in session {session_name}:")
        for video_entry in videos:
            video_basename = os.path.splitext(video_entry.get("file", ""))[0]
            print(f" - {video_basename}")
        return
        
    extracted_dir = os.path.join(args.directory, "extracted")
    if not os.path.exists(extracted_dir):
        try:
            os.makedirs(extracted_dir)
            print(f"Created extraction directory: {extracted_dir}")
        except OSError as e:
            print(f"Error creating directory {extracted_dir}: {e}")
            sys.exit(1)

    # Process videos sequentially, but each video uses parallel extraction
    for video_entry in videos:
        video_basename = os.path.splitext(video_entry.get("file", ""))[0]

        json_file = os.path.join(args.directory, f"{video_basename}.json")
        if os.path.exists(json_file):
            extract_frames(json_file, extracted_dir, args.format, num_jobs)
        else:
            print(f"Warning: JSON file not found for video: {video_basename}")
            

    # Process ROS bag if present
    rosbag_entry = index_data.get("rosbag")
    if rosbag_entry and isinstance(rosbag_entry, dict):
        rosbag_name = rosbag_entry.get("name")
        if rosbag_name:
            bag_path = os.path.join(args.directory, rosbag_name)
            if os.path.exists(bag_path):
                rosbag_to_csv(bag_path, extracted_dir)
            else:
                print(f"Warning: Rosbag directory not found at {bag_path}")

if __name__ == "__main__":
    main()
