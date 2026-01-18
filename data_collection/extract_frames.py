#!/usr/bin/env python3
import cv2
import json
import os
import argparse
import sys
import threading
import csv
import rosbag2_py
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

def rosbag_to_csv(bag_path, output_dir):
    print(f"Converting ROS bag {bag_path} to CSV...")
    reader = rosbag2_py.SequentialReader()
    
    # Try mcap then sqlite3
    storage_id = 'mcap'
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
            rosbag2_py.ConverterOptions('', '')
        )
    except:
        storage_id = 'sqlite3'
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
            rosbag2_py.ConverterOptions('', '')
        )

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

def extract_frames(input_path, output_dir=None):
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
        # Assuming video is in the same directory as json
        video_path = os.path.join(os.path.dirname(input_path), video_filename)
        
    # Fallback if video_file key invalid or file not found, try replacing extension
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
    # data already loaded
    timestamps = data.get("timestamps_ns", data.get("timestamps", data.get("timestamps_ms")))
        
    if not timestamps:
        print("Error: No timestamps found in JSON file")
        return

    # Determine if we need to convert units to seconds for the filename
    is_ns = "timestamps_ns" in data
    is_ms = "timestamps_ms" in data and not is_ns

    # Output directory
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{video_basename}] Processing {video_path}")
    print(f"[{video_basename}] Total frames (metadata): {frame_count}, Total timestamps: {len(timestamps)}")
    print(f"[{video_basename}] Output directory: {output_dir}")
    
    # Iterate
    idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if idx < len(timestamps):
            ts = timestamps[idx]
            
            # Use video basename for the image prefix
            # Format timestamp in filename
            if is_ns:
                # If ns, we can keep as integer or convert to sec with many decimals
                # Using seconds with 9 decimals for ns precision
                ts_str = f"{int(ts) / 1e9:.9f}"
            elif is_ms:
                ts_str = f"{float(ts) / 1000.0:.6f}"
            else:
                ts_str = f"{float(ts):.6f}"
            
            # Construct filename: VideoName_Timestamp.png
            image_name = f"{video_basename}_{ts_str}.png"
            image_path = os.path.join(output_dir, image_name)
            
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"[{video_basename}] Saved {saved_count} images...")
        else:
            # If we have more frames than timestamps, we stop or warn?
            if idx == len(timestamps):
                 print(f"\n[{video_basename}] Warning: Video has more frames than timestamps (stopped at frame {idx})")
            break # Stop if we ran out of timestamps
            
        idx += 1

    cap.release()
    print(f"[{video_basename}] Done. Extracted {saved_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from all videos in a session directory using index.json")
    parser.add_argument("-d", "--directory", help="Path to the recording session directory", required=True)
    parser.add_argument("-l", "--list", action="store_true", help="List names of videos in the session but do not extract frames")
    
    args = parser.parse_args()

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

    threads = []
    for video_entry in videos:
        video_basename = os.path.splitext(video_entry.get("file", ""))[0]

        json_file = os.path.join(args.directory, f"{video_basename}.json")
        if os.path.exists(json_file):
            t = threading.Thread(target=extract_frames, args=(json_file, extracted_dir))
            t.start()
            threads.append(t)
        else:
            print(f"Warning: JSON file not found for video: {video_basename}")
            
    for t in threads:
        t.join()

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
