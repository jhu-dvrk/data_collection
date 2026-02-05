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
        clean_str = time_str.replace("'", "").replace('"', "").strip()
        
        # Try parsing as Date Time (new format: YYYY-MM-DD HH:MM:SS.ffffff)
        try:
             # Handle potential OCR errors like double colons or spaces instead of colons
             # Regex to find: YYYY-MM-DD HH:MM:SS.ffffff
             # Matches: 2024-01-22 18:30:45.123 or 2024-01-22 18:30:45
             match = re.search(r'(\d{4}[-.\s]\d{2}[-.\s]\d{2})\s+(\d{2}[:\s]+\d{2}[:\s]+\d{2})(?:\.(\d+))?', clean_str)
             if match:
                 date_part = match.group(1).replace(" ", "-").replace(".", "-")
                 # Normalize time part (replace double colons or spaces with single colon)
                 time_part = re.sub(r'[:\s]+', ':', match.group(2))
                 frac_part = match.group(3) or "0"
                 # strptime %f expects exactly 6 digits. Pad or truncate.
                 frac_part = (frac_part + "000000")[:6]
                 
                 try:
                     dt_str = f"{date_part} {time_part}.{frac_part}"
                     dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                     return dt.timestamp()
                 except ValueError:
                     pass
                     
             # Fallback to pure time format if date is missing (legacy)
             match_time = re.search(r'(\d{2}[:\s]+\d{2}[:\s]+\d{2})(?:\.(\d+))?', clean_str)
             if match_time:
                 time_part = re.sub(r'[:\s]+', ':', match_time.group(1))
                 frac_part = match_time.group(2) or "0"
                 frac_part = (frac_part + "000000")[:6]
                 try:
                     dt = datetime.datetime.strptime(f"{time_part}.{frac_part}", "%H:%M:%S.%f")
                     # This returns seconds from midnight if we don't care about date
                     return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
                 except ValueError:
                     pass
        except Exception:
             pass
    except Exception:
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
    args: (idx, frame_path, file_ts, top_bbox)
    Returns: (idx, ocr_text, ocr_ts, file_ts)
    """
    idx, frame_path, file_ts, top_bbox = args
    
    # Read Image
    img = cv2.imread(frame_path)
    if img is None:
        return idx, None, None, file_ts
    
    h, w = img.shape[:2]
    strip_height = 30
    strip_start_y = h - strip_height
    
    # --- Bottom: Absolute Time (Clock Time) ---
    # top_bbox is relative to the bottom strip (left side)
    if top_bbox:
        tx1, ty1, tx2, ty2 = top_bbox
        roi_top = img[strip_start_y + ty1 : strip_start_y + ty2, tx1 : tx2]
    else:
        roi_top = None
    
    rel_text = ""
    rel_ts = None
    
    if roi_top is not None and roi_top.size > 0:
        roi_top = cv2.resize(roi_top, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray_top = cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY)
        
        # Tesseract prefers black text on white background.
        # If the image is mostly dark, we invert it.
        if np.mean(gray_top) < 127:
            gray_top = cv2.bitwise_not(gray_top)
            
        _, thresh_top = cv2.threshold(gray_top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if pytesseract:
            try:
                # Expect YYYY-MM-DD HH:MM:SS.ffffff
                # Allow spaces and colons clearly in whitelist
                config = '--psm 7 -c tessedit_char_whitelist=0123456789:.- ' 
                rel_text = pytesseract.image_to_string(thresh_top, config=config).strip()
                rel_ts = parse_gst_time(rel_text)
            except Exception:
                pass

    return idx, rel_text, rel_ts, file_ts

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
        
        if img_0 is not None:
            h, w = img_0.shape[:2]
            # The timestamp overlay is in a 30px strip at the bottom
            strip_height = 30
            strip_start_y = h - strip_height
            
            # The absolute timestamp is on the left side
            mid_x = w // 2
            
            # Left side of strip
            img_strip_left = img_0[strip_start_y:, 0:mid_x]
            top_bbox = get_text_bbox(img_strip_left)
            
            if top_bbox:
                print(f"  Detected Absolute Time BBox (bottom-left): {top_bbox}")
                # Save ROI for debugging
                tx1, ty1, tx2, ty2 = top_bbox
                debug_roi = img_strip_left[ty1:ty2, tx1:tx2]
                debug_path = "debug_roi.png"
                cv2.imwrite(debug_path, debug_roi)
                print(f"  Saved debug ROI to {os.path.abspath(debug_path)}")
            else:
                print("  Warning: Could not detect absolute time in bottom-left strip.")
                # Save the whole strip if detection fails
                cv2.imwrite("debug_strip.png", img_strip_left)
                print(f"  Saved failed strip to {os.path.abspath('debug_strip.png')}")

        # Prepare arguments for parallel processing
        tasks = []
        for idx, frame_ctx in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_ctx['file'])
            file_ts = frame_ctx['ts']
            tasks.append((idx, frame_path, file_ts, top_bbox))

        # Process frames in parallel
        # Results will be a list of (idx, ocr_text, ocr_ts, file_ts)
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

        # Analyze differences sequentially
        diffs = []
        counts = {'total': len(results), 'failed': 0, 'outlier': 0, 'valid': 0}
        
        for result in results:
            idx, text, ocr_ts, file_ts = result
            
            if ocr_ts is None:
                counts['failed'] += 1
                continue
            
            # Reconstruction of full timestamp
            # If ocr_ts is small (seconds from midnight), reconstruct using file date.
            if ocr_ts < 200000: # Arbitrary cutoff, 86400 is max seconds in day
                dt_file = datetime.datetime.fromtimestamp(file_ts)
                day_start = dt_file.replace(hour=0, minute=0, second=0, microsecond=0)
                ocr_dt = day_start + datetime.timedelta(seconds=ocr_ts)
                full_ocr_ts = ocr_dt.timestamp()

                # Handle day wrapping
                diff_check = full_ocr_ts - file_ts
                if diff_check > 12 * 3600:
                    full_ocr_ts -= 24 * 3600
                elif diff_check < -12 * 3600:
                    full_ocr_ts += 24 * 3600
                
                ocr_ts = full_ocr_ts

            # Calculate difference. 
            diff = file_ts - ocr_ts
            
            # Sanity check: If OCR is more than 10s away from file TS, it's likely a misread digit
            if abs(diff) > 10.0:
                 counts['outlier'] += 1
                 continue

            diffs.append(diff)
            counts['valid'] += 1
            
            # Just print a few samples to show it worked
            if idx % 20 == 0:
                 print(f"Frame {idx}: OCR={text} ({ocr_ts:.3f}), FileTS={file_ts:.3f}, Latency={diff*1000:.1f}ms")

        # Summary
        print(f"\nAnalysis Summary for {stream_name}:")
        print(f"  Frames Processed: {counts['total']}")
        print(f"  Valid Timestamps: {counts['valid']} ({(counts['valid']/counts['total'])*100.0:.1f}%)")
        print(f"  Failed OCR:       {counts['failed']}")
        print(f"  Outliers:         {counts['outlier']}")

        if diffs:
             avg_diff = np.mean(diffs)
             std_diff = np.std(diffs)
             print(f"  Average Latency:  {avg_diff*1000:.2f} ms")
             print(f"  Jitter (StdDev):  {std_diff*1000:.2f} ms")
             print(f"  Min Latency:      {np.min(diffs)*1000:.2f} ms")
             print(f"  Max Latency:      {np.max(diffs)*1000:.2f} ms")
        else:
             print("  Warning: No valid timestamps found for latency analysis.")

def main():
    parser = argparse.ArgumentParser(description="Check synchronization of extracted frames using OCR")
    parser.add_argument("-d", "--dir", dest="frames_dir", required=True, help="Directory containing extracted frames (named with timestamps)")
    parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs (default: half of available cores)")
    args = parser.parse_args()
    
    verify_frames(args.frames_dir, num_jobs=args.jobs)

if __name__ == "__main__":
    main()
