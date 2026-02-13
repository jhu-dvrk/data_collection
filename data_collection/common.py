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
        int: The cpu timestamp. Returns 0 if invalid.
    """
    if isinstance(value, dict):
        return int(value.get("cpu_ts", 0))
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
