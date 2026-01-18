#!/usr/bin/env python3
import os
import json
import re
import cv2
import argparse
from datetime import datetime

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
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except Exception:
        pass
    return None

def verify_frames(frames_dir):
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
            ts_sec = float(match.group(2))
            if stream_name not in streams:
                streams[stream_name] = []
            streams[stream_name].append({'filename': f, 'ts_sec': ts_sec})

    if not streams:
        print(f"Could not parse timestamps from filenames in {frames_dir}")
        return

    for stream_name, frames in streams.items():
        print(f"\n--- Analyzing stream: {stream_name} ({len(frames)} frames) ---")
        results = []
        first_ts_sec = frames[0]['ts_sec']
        processed_count = 0
        
        for frame_info in frames:
            filename = frame_info['filename']
            ts_sec = frame_info['ts_sec']
            processed_count += 1
            
            # Relative time according to filename (seconds)
            relative_filename_sec = ts_sec - first_ts_sec
            
            # Read image
            img_path = os.path.join(frames_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Crop to bottom right where timeoverlay is
            h, w = img.shape[:2]
            crop = img[int(h*0.8):, int(w*0.5):]
            
            # OCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
            
            # Clean up common OCR errors
            text = text.replace(' ', '').replace('O', '0').replace('I', '1')
            # Match HH:MM:SS format
            time_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', text)
            
            if time_match:
                ocr_time_str = time_match.group(1)
                ocr_sec = parse_gst_time(ocr_time_str)
                
                if ocr_sec is not None:
                    if len(results) == 0:
                        ocr_offset = ocr_sec
                        print(f"  [Progress] First timestamp found: {ocr_time_str}")
                    
                    expected_ocr_sec = relative_filename_sec + ocr_offset
                    # Check drift (within seconds, as OCR doesn't have nanos now)
                    diff = abs(ocr_sec - expected_ocr_sec)
                    
                    results.append({
                        'diff': diff,
                    })

                    if len(results) % 10 == 0:
                        print(f"  [Progress] Processed {processed_count}/{len(frames)} frames. Last OCR timestamp: {ocr_time_str}")
        
        if not results:
            print(f"Skipping {stream_name}: No frames could be successfully OCR'd.")
            continue

        max_diff = max(r['diff'] for r in results)
        avg_diff = sum(r['diff'] for r in results) / len(results)
        
        print(f"  Verified {len(results)}/{len(frames)} frames.")
        print(f"  Max jitter/error: {max_diff*1000:.2f} ms")
        print(f"  Avg error: {avg_diff*1000:.2f} ms")
        
        if max_diff < 0.05:
            print(f"  SUCCESS: {stream_name} timestamps match video content.")
        else:
            print(f"  WARNING: {stream_name} has significant discrepancy.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check sidecar timestamps against burned-in video time.")
    parser.add_argument("dir", help="Directory containing extracted PNG frames")
    args = parser.parse_args()
    
    verify_frames(args.dir)
