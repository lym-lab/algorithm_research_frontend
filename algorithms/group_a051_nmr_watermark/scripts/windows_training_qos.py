"""Opt this training process out of automatic Windows execution throttling."""
import ctypes
import os
from ctypes import wintypes


class PowerThrottlingState(ctypes.Structure):
    _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                ('StateMask', wintypes.ULONG)]


def enable_training_performance():
    if os.name != 'nt':
        return {'supported': False}
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    signature = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetProcessInformation.argtypes = signature
    kernel.SetProcessInformation.argtypes = signature
    process = kernel.GetCurrentProcess()
    state = PowerThrottlingState(1, 0, 0)
    if not kernel.GetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Explicit HighQoS; preserve unrelated policies such as timer resolution.
    state.ControlMask |= 1
    state.StateMask &= ~1
    if not kernel.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not state.ControlMask & 1 or state.StateMask & 1:
        raise OSError('Windows did not retain the requested training performance policy')
    return {'supported': True, 'execution_throttling': False, 'pid': os.getpid()}
