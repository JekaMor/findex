"""Peak memory of the current process, for the benchmark tables."""

import ctypes
import sys


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [  # ctypes reads this class attribute; it is the layout
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_rss() -> int:
    """Peak resident set size of this process in bytes (Windows: peak working set)."""
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo(
            ctypes.c_void_p(kernel32.GetCurrentProcess()),
            ctypes.byref(counters),
            counters.cb,
        )
        peak: int = counters.PeakWorkingSetSize
        return peak
    import resource

    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib if sys.platform == "darwin" else kib * 1024
