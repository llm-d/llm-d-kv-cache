"""Shared access-age eligibility for discovery and deletion."""


def is_recently_accessed(atime: float, now: float, threshold_seconds: float) -> bool:
    return now - atime < threshold_seconds
