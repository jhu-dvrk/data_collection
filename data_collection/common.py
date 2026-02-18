from datetime import datetime
import time

def get_current_timestamp_iso8601(dt=None):
    """
    Returns the current time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sss)
    If no dt is provided, uses current time.
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def parse_stage_timestamp(value):
    """
    Parses a stage timestamp from a JSON value.
    Handles both legacy integer format and new object format.
    
    Args:
        value: The value associated with "start" or "end" in a stage object.
               Can be an integer (legacy) or a dict {"cpu_ts": ..., "generated_at": ...}
    
    Returns:
        tuple: (cpu_ts, frame). Either or both can be None. Returns (0, None) if invalid.
    """
    if isinstance(value, dict):
        return value.get("cpu_ts"), value.get("frame")
    try:
        return int(value), None
    except (ValueError, TypeError):
        return 0, None
