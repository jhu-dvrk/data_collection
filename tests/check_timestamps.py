#!/usr/bin/env python3
import os
import re
import cv2
import argparse
import datetime
import numpy as np
import multiprocessing
from functools import partial

try:
    import pytesseract
except ImportError:
    pytesseract = None

def parse_gst_time(time_str):
    """Parses GStreamer time format into seconds."""
    if not time_str:
        return None
        
    try:
        # Cleanup: Remove extra spaces
        time_str = time_str.strip()
        
        # Aggressively strip quotes from start/end or inside
        # Tesseract might read '10:00:00' as '10:00:00 (missing one quote) or 10:00:00'
        # or even put spaces like ' 10:00:00 '
        clean_str = time_str.replace("'", "").replace('"', "").strip()
        
        # Check if it is a 9-digit nanosecond string (%N)
        if len(clean_str) == 9 and clean_str.isdigit():
            return int(clean_str) / 1e9

        # Try parsing as Date Time (new clockoverlay format: YYYY-MM-DD HH:MM:SS)
        try:
             dt = datetime.datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
             return dt.timestamp()
        except ValueError:
             pass

        parts = clean_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            
            # Handle seconds
            if '.' in parts[2]:
                sec_parts = parts[2].split('.')
                seconds = int(sec_parts[0])
                ns_str = sec_parts[1]
                # Keep digits only for fraction
                ns_str = ''.join(filter(str.isdigit, ns_str))
                ns = float("0." + ns_str) if ns_str else 0.0
                return hours * 3600 + minutes * 60 + seconds + ns
            else:
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
    except Exception:
        # print(f"Debug: Failed to parse '{time_str}'")
        pass
    return None

def get_text_bbox(img_bgr):
    """
    Finds the bounding box (x1, y1, x2, y2) of white text on black background.
    """
    if img_bgr is None:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Basic threshold to find white pixels
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return None
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    pad = 5
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    
    return (x1, y1, x2, y2)

def process_single_frame(args):
    """
    Worker function to process a single frame.
    args: (idx, frame_path, file_ts, top_bbox, bottom_bbox)
    Returns: (idx, rel_text, rel_ts, abs_text, abs_ts, file_ts)
    """
    idx, frame_path, file_ts, top_bbox, bottom_bbox = args
    
    # Read Image
    img = cv2.imread(frame_path)
    if img is None:
        return idx, None, None, None, None, file_ts
    
    h, w = img.shape[:2]
    mid_y = h // 2
    
    # --- Top Half: Relative Time (Running Time) ---
    # Top bbox is relative to Top Half
    if top_bbox:
        tx1, ty1, tx2, ty2 = top_bbox
        # Slicing: img[:mid_y][ty1:ty2, tx1:tx2] is correct
        roi_top = img[ty1:ty2, tx1:tx2]
    else:
        # Fallback if no bbox provided
        roi_top = None
    
    rel_text = ""
    rel_ts = None
    
    if roi_top is not None and roi_top.size > 0:
        roi_top = cv2.resize(roi_top, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray_top = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
        gray_top = cv2.bitwise_not(gray_top)
        _, thresh_top = cv2.threshold(gray_top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if pytesseract:
            try:
                # Expect digits, colons, dots. No quotes.
                config = '--psm 6 -c tessedit_char_whitelist=0123456789:.' 
                rel_text = pytesseract.image_to_string(thresh_top, config=config).strip()
                rel_ts = parse_gst_time(rel_text)
            except Exception:
                pass

    # --- Bottom Half: Absolute Time (Clock Time) ---
    # Bottom bbox is relative to Bottom Half
    if bottom_bbox:
        bx1, by1, bx2, by2 = bottom_bbox
        # Absolute Y in image is mid_y + by1
        roi_bottom = img[mid_y+by1:mid_y+by2, bx1:bx2]
    else:
        roi_bottom = None
    
    abs_text = ""
    abs_ts = None
    
    if roi_bottom is not None and roi_bottom.size > 0:
        roi_bottom = cv2.resize(roi_bottom, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray_bottom = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
        gray_bottom = cv2.bitwise_not(gray_bottom)
        _, thresh_bottom = cv2.threshold(gray_bottom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if pytesseract:
            try:
                # Use PSM 7 (Single Line) given we have a tight crop.
                # Remove whitelist to see raw output (debug mode)
                config = '--psm 7' 
                abs_text = pytesseract.image_to_string(thresh_bottom, config=config).strip()
                abs_ts = parse_gst_time(abs_text)
            except Exception as e:
                # Print output to debug why OCR is failing (e.g. TesseractNotFoundError)
                print(f"Frame {idx}: OCR Error: {e}")
                pass

    return idx, rel_text, rel_ts, abs_text, abs_ts, file_ts

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
        
        # Determine ROI from first frame to speed up and stabilize OCR
        first_frame_path = os.path.join(frames_dir, frames[0]['file'])
        img_0 = cv2.imread(first_frame_path)
        top_bbox = None
        bottom_bbox = None
        
        if img_0 is not None:
            h, w = img_0.shape[:2]
            mid_y = h // 2
            
            # Top half
            img_top = img_0[0:mid_y, :]
            top_bbox = get_text_bbox(img_top)
            
            # Bottom half
            img_bottom = img_0[mid_y:, :]
            bottom_bbox = get_text_bbox(img_bottom)
            
            if top_bbox:
                print(f"  Detected Top BBox: {top_bbox}")
            else:
                print("  Warning: Could not detect text in top half of first frame.")

            if bottom_bbox:
                print(f"  Detected Bottom BBox: {bottom_bbox}")
            else:
                print("  Warning: Could not detect text in bottom half of first frame.")

        # Prepare arguments for parallel processing
        tasks = []
        for idx, frame_ctx in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_ctx['file'])
            file_ts = frame_ctx['ts']
            tasks.append((idx, frame_path, file_ts, top_bbox, bottom_bbox))

        # Process frames in parallel
        # Results will be a list of (idx, rel_text, rel_ts, abs_text, abs_ts, file_ts)
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
        prev_abs_time = None
        prev_result = None

        for result in results:
            idx, rel_text, rel_ts, abs_text, abs_ts, file_ts = result
            
            # Using Absolute Time (Bottom Overlay) for Wall Clock verification
            ocr_ts = abs_ts
            text = abs_text
            
            if ocr_ts is None:
                # Print failure for every frame to debug
                print(f"Frame {idx}: Failed to parse AbsTimestamp. raw_ocr_output='{text}'") 
                continue
            
            # Reconstruction of full timestamp
            # If ocr_ts is small (seconds from midnight), reconstruct using file date.
            # If ocr_ts is large (full timestamp), use directly.
            
            if ocr_ts < 200000: # Arbitrary cutoff, 86400 is max seconds in day
                dt_file = datetime.datetime.fromtimestamp(file_ts)
                day_start = dt_file.replace(hour=0, minute=0, second=0, microsecond=0)
                ocr_dt = day_start + datetime.timedelta(seconds=ocr_ts)
                full_ocr_ts = ocr_dt.timestamp()

                # Handle day wrapping (e.g. file_ts is 00:00:01, ocr reads 23:59:59 from previous day)
                diff_check = full_ocr_ts - file_ts
                if diff_check > 12 * 3600:
                    full_ocr_ts -= 24 * 3600
                elif diff_check < -12 * 3600:
                    full_ocr_ts += 24 * 3600
                
                ocr_ts = full_ocr_ts

            # Calculate difference. 
            diff = file_ts - ocr_ts
            
            # Detect Transition: When OCR integer second changes.
            if prev_abs_time is not None:
                if ocr_ts != prev_abs_time:
                    # Time changed!
                    if ocr_ts == prev_abs_time + 1.0:
                        # Smooth 1 second increment
                        prev_idx, _, _, _, _, prev_file_ts = prev_result
                        
                        transition_entry = {
                            'frame': idx,
                            'prev_file_ts': prev_file_ts,
                            'curr_file_ts': file_ts,
                            'ocr_ts': ocr_ts,
                            'delta': file_ts - ocr_ts 
                        }
                        transitions.append(transition_entry)

            diffs.append(diff)
            prev_abs_time = ocr_ts
            prev_result = result
            
            # Just print a few samples to show it worked
            if idx % 20 == 0:
                 print(f"Frame {idx}: Abs={text} ({ocr_ts:.3f}), Rel={rel_text} ({rel_ts if rel_ts else 'None'}), FileTS={file_ts:.3f}, Diff={diff:.3f}")

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
