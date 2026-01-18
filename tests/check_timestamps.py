#!/usr/bin/env python3
import os
import re
import cv2
import argparse
import numpy as np
import multiprocessing
from functools import partial

try:
    import pytesseract
except ImportError:
    pytesseract = None

def parse_gst_time(time_str):
    """Parses GStreamer time format into seconds."""
    try:
        # Check if it is a 9-digit nanosecond string (%N)
        if len(time_str) == 9 and time_str.isdigit():
            return int(time_str) / 1e9
        
        # GStreamer timeoverlay/clockoverlay format: HH:MM:SS.nnnnnnnnn
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            if '.' in parts[2]:
                sec_parts = parts[2].split('.')
                seconds = int(sec_parts[0])
                ns = float("0." + sec_parts[1])
                return hours * 3600 + minutes * 60 + seconds + ns
            else:
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds
    except Exception:
        pass
    return None

def process_single_frame(args):
    """
    Worker function to process a single frame.
    args: (idx, frame_path, file_ts)
    Returns: (idx, text, ocr_ts, file_ts)
    """
    idx, frame_path, file_ts = args
    
    # Read Image
    img = cv2.imread(frame_path)
    if img is None:
        return idx, None, None, file_ts
    
    # Crop to bottom-right quadrant to isolate clockoverlay (avoiding timeoverlay at top-left)
    h, w = img.shape[:2]
    # clockoverlay is bottom-right. Taking bottom 25% and right 50% should be safe and sufficient
    roi = img[int(h*0.75):h, int(w*0.5):w]

    # Preprocess for OCR (grayscale, threshold)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Thresholding to extract white text from potentially colored background
    # Assuming clockoverlay is white text. 
    # Otsu's thresholding works well for high contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    text = ""
    ocr_ts = None
    
    if pytesseract:
        # Use PSM 7 (Treat the image as a single text line)
        try:
            text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
            ocr_ts = parse_gst_time(text)
        except Exception:
            pass

    return idx, text, ocr_ts, file_ts

def verify_frames(frames_dir, num_jobs=None):
    if pytesseract is None:
        print("Error: 'pytesseract' not found. Please install it: pip install pytesseract")
        print("You also need the tesseract-ocr system package: sudo apt install tesseract-ocr")
        return

    # Find all PNG files in the directory
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    frame_files.sort()

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return

    # Group files by video stream name
    streams = {}
    for f in frame_files:
        # Extract video name and timestamp (e.g., test_stream_1234.5678.png)
        match = re.search(r'^(.*)_(\d+\.\d+)\.png$', f)
        if match:
            stream_name = match.group(1)
            ts = float(match.group(2))
            if stream_name not in streams:
                streams[stream_name] = []
            streams[stream_name].append({'file': f, 'ts': ts})

    print(f"Found {len(streams)} streams.")
    
    # Determine number of processes to use
    cpu_count = os.cpu_count() or 1
    if num_jobs is None:
        num_processes = max(1, cpu_count // 2)
    else:
        num_processes = int(num_jobs)
        
    if num_processes < 1:
        num_processes = 1

    print(f"Using {num_processes} processes for OCR (Total CPUs: {cpu_count}).")

    for stream_name, frames in streams.items():
        # Sort by timestamp
        frames.sort(key=lambda x: x['ts'])
        print(f"\nChecking stream: {stream_name} ({len(frames)} frames)")
        
        # Prepare arguments for parallel processing
        tasks = []
        for idx, frame_ctx in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_ctx['file'])
            file_ts = frame_ctx['ts']
            tasks.append((idx, frame_path, file_ts))

        # Process frames in parallel
        # Results will be a list of (idx, text, ocr_ts, file_ts)
        results = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            # chunksize can help performance for many small tasks
            chunksize = max(1, len(tasks) // (num_processes * 4))
            for res in pool.imap(process_single_frame, tasks, chunksize=chunksize):
                results.append(res)
                if len(results) % 50 == 0:
                    print(f"Processed {len(results)}/{len(tasks)} frames...", end='\r')
        print(f"Processed {len(results)}/{len(tasks)} frames. Done.")

        # Sort results by index to ensure correct order
        results.sort(key=lambda x: x[0])

        # Analyze transitions sequentially
        diffs = []
        transitions = []
        prev_ocr_time = None
        prev_result = None

        for result in results:
            idx, text, ocr_ts, file_ts = result
            
            if ocr_ts is None:
                if idx % 10 == 0:
                    print(f"Frame {idx}: Failed to read timestamp. OCR text: '{text}'")
                continue
            
            # Calculate difference. 
            diff = file_ts - ocr_ts
            
            # Detect Transition: When OCR integer second changes.
            if prev_ocr_time is not None:
                if ocr_ts != prev_ocr_time:
                    # Time changed!
                    if ocr_ts == prev_ocr_time + 1.0:
                        # Smooth 1 second increment
                        prev_idx, _, _, prev_file_ts = prev_result
                        
                        transition_entry = {
                            'frame': idx,
                            'prev_file_ts': prev_file_ts,
                            'curr_file_ts': file_ts,
                            'ocr_ts': ocr_ts,
                            'delta': file_ts - ocr_ts 
                        }
                        transitions.append(transition_entry)

            diffs.append(diff)
            prev_ocr_time = ocr_ts
            prev_result = result
            
            # Just print a few samples to show it worked
            if idx % 20 == 0:
                 print(f"Frame {idx}: OCR={text} ({ocr_ts:.3f}), FileTS={file_ts:.3f}, Diff={diff:.3f}")

        # Summary
        if transitions:
            print(f"\n--- Transition Analysis (Sub-second Estimation) ---")
            deltas = [t['delta'] for t in transitions]
            avg_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            min_delta = np.min(deltas)
            max_delta = np.max(deltas)
            
            print(f"Detected {len(transitions)} second-boundaries.")
            print(f"Average Latency (FileCaptureTime - OCR_Second): {avg_delta*1000:.2f} ms")
            print(f"Jitter (StdDev): {std_delta*1000:.2f} ms")
            print(f"Min Latency: {min_delta*1000:.2f} ms")
            print(f"Max Latency: {max_delta*1000:.2f} ms")
            
            print("\nDetailed Transitions:")
            for t in transitions[:5]: # Show first 5
                 print(f"  Frame {t['frame']}: Changed to {t['ocr_ts']}s. Capture Time: {t['curr_file_ts']:.3f}. Latency: {t['delta']*1000:.1f}ms")
            if len(transitions) > 5:
                print(f"  ... and {len(transitions)-5} more.")

        elif diffs:
             print("\nNo second-boundaries detected (maybe short video?).")
             avg_diff = np.mean(diffs)
             print(f"Average Diff (FileTS - OCR): {avg_diff:.3f} s")

def main():
    parser = argparse.ArgumentParser(description="Check synchronization of extracted frames using OCR")
    parser.add_argument("frames_dir", help="Directory containing extracted frames (named with timestamps)")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs (default: half of available cores)")
    args = parser.parse_args()
    
    verify_frames(args.frames_dir, num_jobs=args.jobs)

if __name__ == "__main__":
    main()
