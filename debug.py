# Code from https://blog.streamlit.io/3-steps-to-fix-app-memory-leaks/

import tracemalloc, json
import streamlit as st
import gc

@st.cache_resource
def init_tracking_object():
  tracemalloc.start(10)

  return {
    "runs": 0,
    "tracebacks": {}
  }



def traceback_exclude_filter(patterns, tracebackList):
    """
    Returns False if any provided pattern exists in the filename of the traceback,
    Returns True otherwise.
    """
    for t in tracebackList:
        for p in patterns:
            if p in t.filename:
                return False
        return True


def traceback_include_filter(patterns, tracebackList):
    """
    Returns True if any provided pattern exists in the filename of the traceback,
    Returns False otherwise.
    """
    for t in tracebackList:
        for p in patterns:
            if p in t.filename:
                return True
    return False


def check_for_leaks(diff,_TRACES):
    """
    Checks if the same traceback appears consistently after multiple runs.

    diff - The object returned by tracemalloc#snapshot.compare_to
    """
    _TRACES["runs"] = _TRACES["runs"] + 1
    tracebacks = set()

    for sd in diff:
        for t in sd.traceback:
            tracebacks.add(t)

    if "tracebacks" not in _TRACES or len(_TRACES["tracebacks"]) == 0:
        for t in tracebacks:
            _TRACES["tracebacks"][t] = 1
    else:
        oldTracebacks = _TRACES["tracebacks"].keys()
        intersection = tracebacks.intersection(oldTracebacks)
        evictions = set()
        for t in _TRACES["tracebacks"]:
            if t not in intersection:
                evictions.add(t)
            else:
                _TRACES["tracebacks"][t] = _TRACES["tracebacks"][t] + 1

        for t in evictions:
            del _TRACES["tracebacks"][t]

    if _TRACES["runs"] > 1:
        st.write(f'After {_TRACES["runs"]} runs the following traces were collected.')
        prettyPrint = {}
        for t in _TRACES["tracebacks"]:
            prettyPrint[str(t)] = _TRACES["tracebacks"][t]
        st.write(json.dumps(prettyPrint, sort_keys=True, indent=4))


def compare_snapshots(_TRACES):
    """
    Compares two consecutive snapshots and tracks if the same traceback can be found
    in the diff. If a traceback consistently appears during runs, it's a good indicator
    for a memory leak.
    """
    snapshot = tracemalloc.take_snapshot()
    if "snapshot" in _TRACES:
        diff = snapshot.compare_to(_TRACES["snapshot"], "lineno")
        diff = [d for d in diff if
                d.count_diff > 0 and traceback_exclude_filter(["tornado"], d.traceback)
                and traceback_include_filter(["streamlit"], d.traceback)
                ]
        check_for_leaks(diff, _TRACES)

    _TRACES["snapshot"] = snapshot