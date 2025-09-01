import os
import time
import ctypes
import getpass
import datetime
import pyautogui
import subprocess
import win32con
import win32ts
import win32security
import win32api
from contextlib import contextmanager


def sleepComputer(hibernate: bool = False, force: bool = True, disable_wake_events_check: bool = False):
    """
    Put the Windows computer into sleep (S3) or hibernate (S4) mode.

    Parameters:
    - hibernate (bool): True → hibernate (S4), False → sleep (S3)
    - force (bool): True → force sleep/hibernate even if applications are blocking it.
                    False → allow applications to prevent sleep/hibernate.
    - disable_wake_events_check (bool): True → ignore wake events (prevents some blockers)
                                        False → check wake events (may delay sleep)

    Notes:
    - The underlying Windows API is `SetSuspendState(Hibernate, Force, DisableWakeCheck)`
      where each parameter is an integer (0 or 1).
    """
    hibernate_flag = int(hibernate)
    force_flag = int(force)
    disable_wake_flag = int(disable_wake_events_check)

    #command = f"rundll32.exe powrprof.dll,SetSuspendState {hibernate_flag},{force_flag},{disable_wake_flag}"
    #command = f"rundll32.exe powrprof.dll,SetSuspendState Sleep"
    # command = f"Add-Type -AssemblyName System.Windows.Forms [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"
    # respons = os.system(command)
    ps = r"""
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)
        """
    respons = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
        capture_output=True, text=True
    )
    return respons == 0

def deleteTask(task_name:str):
    delete_command = f'schtasks /delete /tn "{task_name}" /f >nul 2>&1'
    respons = os.system(delete_command)
    return respons == 0

def enableWake(task_name="WakeUpTask"):
    ps_command = f"""
    $task = Get-ScheduledTask -TaskName '{task_name}';
    $settings = $task.Settings;
    $settings.WakeToRun = $true;
    Set-ScheduledTask -TaskName '{task_name}' -Settings $settings
    """
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps_command], capture_output=True, text=True)
    print(result.stdout, result.stderr)
    return result.returncode == 0

def wakeUpScreen():
    # Move mouse slightly to wake monitor
    ctypes.windll.user32.mouse_event(0x0001, 1, 0, 0, 0)
    ctypes.windll.user32.mouse_event(0x0001, -1, 0, 0, 0)
    
def scheduleWakeUpIn(
    weeks:float=0,
    days:float=0, 
    hours:float=0, 
    minutes:float=0, 
    seconds:float=0, 
    microseconds:float=0, 
    milliseconds:float=0, 
    task_name: str = "WakeUpTask",
    user: str = None,
    run_as_system: bool = True,
    command_to_run: str = 'cmd.exe /c echo wakeUp',
    highest_privileges: bool = True
):
    """
    Schedule a Windows task to run once at a specific time in the future.

    Parameters:
    - weeks (float): Delay from now in weeks
    - days (float): Delay from now in days
    - hours (float): Delay from now in hours
    - minutes (float): Delay from now in minutes
    - seconds (float): Delay from now in seconds
    - microseconds (float): Delay from now in microseconds
    - milliseconds (float): Delay from now in milliseconds
    - task_name (str): Name of the scheduled task
    - user (str): Windows user to run the task as (ignored if run_as_system=True)
    - run_as_system (bool): Run as SYSTEM account if True, otherwise as specified user
    - command_to_run (str): Command executed by the task
    - highest_privileges (bool): Run task with highest privileges

    Behavior:
    - Automatically calculates the target time (HH:MM) and date (DD/MM/YYYY)
    - Adjusts date if the target time passes midnight
    - Deletes any previous task with the same name
    - Creates a new task with the specified parameters

    Notes on schtasks parameters:
    - /SC once -> run task once
    - /TN <task_name> -> task name
    - /TR <command_to_run> -> command executed
    - /ST <HH:MM> -> start time
    - /SD <DD/MM/YYYY> -> start date
    - /RU <user> -> run as user/SYSTEM
    - /RL HIGHEST -> run with highest privileges
    - /F -> force overwrite if task exists
    """
    now = datetime.datetime.now()
    target_time = now + datetime.timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds, milliseconds=milliseconds)

    wake_time_str = target_time.strftime("%H:%M")

    # Determine correct target date
    target_date_str = target_time.strftime("%d/%m/%Y")

    # Determine which user to run as
    
    run_user = "SYSTEM" if run_as_system else user or getpass.getuser()

    # Delete existing task if any
    deleteTask(task_name)

    # Build schtasks command
    command_parts = [
        "schtasks",
        "/create",
        f'/sc once',
        f'/tn "{task_name}"',
        f'/tr "{command_to_run}"',
        f'/st {wake_time_str}',
        f'/sd {target_date_str}',
        f'/ru {run_user}',
        "/f"
    ]
    
    if highest_privileges:
        command_parts.append("/rl HIGHEST")

    command = " ".join(command_parts)
    respons = os.system(command)
    wakeup = enableWake(task_name=task_name)

    return {
        'status':respons == 0 and wakeup,
        'date':target_date_str,
        'hour':wake_time_str
    }

@contextmanager
def noSleep(display: bool = True):
    """
    Prevents Windows from going into standby mode while the `with` block is running.

    Args:
        display (bool): If True, also prevents the screen from turning off.
        
    Exemple:
        with noSleep(display=True):
            # my code
    """
    # Constant values
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    
    # Include the screen sleep ?
    if display:
        flags |= ES_DISPLAY_REQUIRED

    # activate "no sleep"
    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    try:
        yield
    finally:
        # Restauration of the normal behavior
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)