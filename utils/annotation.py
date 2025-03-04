from typing import List, Dict, Union


def search_stage(current_timestamp:float, stages:List[Dict[str, Union[float, list, str]]]) -> int:
    current_stage_idx = 0
    for stage_idx in range(len(stages)):
        if stages[stage_idx]['timestamp_ms'] <= current_timestamp and stages[stage_idx]['timestamp_ms'] >= stages[current_stage_idx]['timestamp_ms']:
            current_stage_idx = stage_idx
    return current_stage_idx
