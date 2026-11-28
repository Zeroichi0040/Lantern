"""Windows kill-on-close job for an owned inference process."""
import os

if os.name == "nt":
    import ctypes as C
    from ctypes import wintypes as W

    class BasicLimits(C.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", C.c_int64),
            ("PerJobUserTimeLimit", C.c_int64),
            ("LimitFlags", W.DWORD),
            ("MinimumWorkingSetSize", C.c_size_t),
            ("MaximumWorkingSetSize", C.c_size_t),
            ("ActiveProcessLimit", W.DWORD),
            ("Affinity", C.c_size_t),
            ("PriorityClass", W.DWORD),
            ("SchedulingClass", W.DWORD),
        ]

    class IOCounters(C.Structure):
        _fields_ = [(name, C.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(C.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits), ("IoInfo", IOCounters),
            ("ProcessMemoryLimit", C.c_size_t), ("JobMemoryLimit", C.c_size_t),
            ("PeakProcessMemoryUsed", C.c_size_t), ("PeakJobMemoryUsed", C.c_size_t),
        ]

    kernel = C.WinDLL("kernel32", use_last_error=True)
    def bind(name, result, *args):
        fn = getattr(kernel, name)
        fn.restype, fn.argtypes = result, args
        return fn

    create_job = bind("CreateJobObjectW", W.HANDLE, C.c_void_p, W.LPCWSTR)
    set_info = bind("SetInformationJobObject", W.BOOL, W.HANDLE, C.c_int, C.c_void_p, W.DWORD)
    open_process = bind("OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    assign_job = bind("AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
    close_handle = bind("CloseHandle", W.BOOL, W.HANDLE)


class WindowsJob:
    """Non-inheritable job handle; Windows kills its members when Lantern exits."""
    def __init__(self):
        self.handle = create_job(None, None)
        if not self.handle:
            raise C.WinError(C.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not set_info(self.handle, 9, C.byref(limits), C.sizeof(limits)):
            error = C.WinError(C.get_last_error())
            self.close()
            raise error

    def assign(self, pid):
        process = open_process(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not process:
            raise C.WinError(C.get_last_error())
        try:
            if not assign_job(self.handle, process):
                raise C.WinError(C.get_last_error())
        finally:
            close_handle(process)

    def close(self):
        if self.handle:
            if not close_handle(self.handle):
                raise C.WinError(C.get_last_error())
            self.handle = None
