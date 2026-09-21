"""Joint allocation of a KV bit budget across retention, precision and protection (M5)."""

from raq_kv.allocation.policy import Allocation, allocate, single_signal

__all__ = ["Allocation", "allocate", "single_signal"]
