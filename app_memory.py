"""Pressure-triggered working-set trim. Never terminates or suspends an app.

Only same-user desktop apps with all visible windows minimized, verified quiet
process families and a sustained RAM shortage for a real model task qualify.
Working-set reduction is not commit/VRAM release or a guaranteed system gain.
"""
import copy
import ctypes
from ctypes import wintypes
import math
import ntpath
import os
import threading
import time
from collections import deque

import psutil

GIB = 1024 ** 3
MIB = 1024 ** 2
INTERVAL = 5.0
IDLE_SECONDS = 600.0
PRESSURE_SECONDS = 15.0
COOLDOWN_SECONDS = 1800.0
ACTION_INTERVAL = 60.0
MIN_RSS = 256 * MIB
MAX_PROCESSES = 4
MAX_EPISODE_ACTIONS = 3
CPU_LIMIT = .5  # Percent of one logical CPU, aggregated over the app family.
IO_LIMIT = 4096.0  # Bytes/sec, including other I/O (not just disk).

# Keep work, inference, recording, editing and interactive agents untouched.
# Unknown/no-window services are excluded independently of this name defense.
PROTECTED_NAMES = set('''system registry memory compression secure system
    explorer.exe dwm.exe taskmgr.exe chatgpt.exe codex.exe codex-app.exe
    grok bot.exe python.exe pythonw.exe node.exe java.exe javaw.exe
    cmd.exe powershell.exe pwsh.exe conhost.exe windowsterminal.exe
    wsl.exe wslhost.exe wslservice.exe vmmem.exe vmware-vmx.exe virtualboxvm.exe
    ffmpeg.exe ffplay.exe obs64.exe obs32.exe obs.exe blender.exe
    winword.exe excel.exe powerpnt.exe outlook.exe onenote.exe
    wps.exe et.exe wpp.exe notepad.exe notepad++.exe code.exe devenv.exe
    photoshop.exe afterfx.exe resolve.exe capcut.exe jianyingpro.exe
    vlc.exe potplayermini64.exe potplayermini.exe cloudmusic.exe qqmusic.exe
    audacity.exe claude.exe opencode.exe cursor.exe'''.split()) | {
        'grok bot.exe', 'deepseek harness.exe', 'adobe premiere pro.exe', '哔哩哔哩.exe'}


def path_key(path):
    return ntpath.normcase(ntpath.normpath(path or ''))


def within(path, root):
    path, root = path_key(path), path_key(root)
    return bool(root and root != '.') and (path == root or path.startswith(root.rstrip('\\') + '\\'))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity_key(process):
    return (process['pid'], process['created'], path_key(process['exe']))


def signature(app):
    return tuple(sorted(identity_key(p) for p in app['processes']))


def quiet_delta(old, new, elapsed, immediate=False):
    if elapsed <= 0 or signature(old) != signature(new):
        return False
    cpu = new['cpu_seconds'] - old['cpu_seconds']
    io = new['io_bytes'] - old['io_bytes']
    # Native counters have finite precision during the final sub-second check.
    return (0 <= cpu <= max(.02 if immediate else 0, elapsed * CPU_LIMIT / 100)
            and 0 <= io <= max(65536 if immediate else 0, elapsed * IO_LIMIT))


class IdleTracker:
    """Monotonic, bounded history; gaps, activity or identity changes reset it."""
    def __init__(self):
        self.rows = {}

    def observe(self, apps, now):
        rows, result = {}, []
        for app in apps[:128]:
            key = path_key(app['exe'])
            old = self.rows.get(key)
            quiet = (app['safe'] and old and old['app']['safe'] and
                     0 < now-old['time'] <= INTERVAL*3 and
                     quiet_delta(old['app'], app, now-old['time']))
            since = old['since'] if quiet else now
            rows[key] = {'app': app, 'time': now, 'since': since}
            item = dict(app, idle_seconds=max(0, now-since), sampled_at=now)
            item['eligible'] = bool(app['safe'] and item['idle_seconds'] >= IDLE_SECONDS
                                    and app['rss'] >= MIN_RSS)
            result.append(item)
        self.rows = rows
        return result


class WindowsApps:
    """Read-only inventory plus the one permitted native write: EmptyWorkingSet."""
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows desktop is required')
        self.k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self.u32 = ctypes.WinDLL('user32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.enum_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        specs = [
            (self.u32.GetForegroundWindow, [], wintypes.HWND),
            (self.u32.GetWindowThreadProcessId, [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            (self.u32.EnumWindows, [self.enum_callback, wintypes.LPARAM], wintypes.BOOL),
            (self.u32.IsWindowVisible, [wintypes.HWND], wintypes.BOOL),
            (self.u32.IsIconic, [wintypes.HWND], wintypes.BOOL),
            (self.k32.ProcessIdToSessionId, [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            (self.k32.OpenProcess, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            (self.k32.CloseHandle, [wintypes.HANDLE], wintypes.BOOL),
            (self.k32.GetProcessTimes, [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4, wintypes.BOOL),
            (self.k32.QueryFullProcessImageNameW, [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            (self.k32.GetExitCodeProcess, [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            (self.psapi.EmptyWorkingSet, [wintypes.HANDLE], wintypes.BOOL),
        ]
        for function, args, result in specs:
            function.argtypes, function.restype = args, result
        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                       ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                       ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                       ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                       ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR*260)]
        self.process_entry = PROCESSENTRY32W
        self.k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for function in (self.k32.Process32FirstW, self.k32.Process32NextW):
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
            function.restype = wintypes.BOOL
        self.username = psutil.Process().username().casefold()
        self.session = self.session_id(os.getpid())
        self.windows_root = os.environ.get('SystemRoot', 'C:\\Windows')

    def session_id(self, pid):
        result = wintypes.DWORD()
        if not self.k32.ProcessIdToSessionId(pid, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return result.value

    def foreground(self):
        hwnd = self.u32.GetForegroundWindow()
        pid = wintypes.DWORD()
        if not hwnd or not self.u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
            raise OSError('Foreground desktop is unavailable')
        return pid.value

    def windows(self):
        result = {}
        @self.enum_callback
        def collect(hwnd, _):
            if self.u32.IsWindowVisible(hwnd):
                pid = wintypes.DWORD()
                if self.u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
                    result.setdefault(pid.value, []).append(bool(self.u32.IsIconic(hwnd)))
            return True
        if not self.u32.EnumWindows(collect, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    @staticmethod
    def expand_descendants(pids, processes):
        result = set(pids)
        while True:
            children = {pid for pid, p in processes.items() if p.get('ppid') in result}
            if children <= result:
                return result
            result |= children

    def processes(self):
        # One native snapshot supplies PID/parent/name. Opening every system
        # process for exe/username can take seconds; inspect only GUI candidates.
        handle = self.k32.CreateToolhelp32Snapshot(2, 0)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = self.process_entry()
            entry.dwSize = ctypes.sizeof(entry)
            result = {}
            ok = self.k32.Process32FirstW(handle, ctypes.byref(entry))
            while ok:
                result[entry.th32ProcessID] = {'pid': entry.th32ProcessID,
                    'ppid': entry.th32ParentProcessID, 'name': entry.szExeFile}
                ok = self.k32.Process32NextW(handle, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:
                raise ctypes.WinError(ctypes.get_last_error())
            return result
        finally:
            self.k32.CloseHandle(handle)

    def snapshot(self, context):
        windows, foreground = self.windows(), self.foreground()
        processes = self.processes()
        protected = set(context.get('protected_pids', [])) | {os.getpid(), foreground}
        # Helpers may own their own windows (Office WebView, agents, recorders).
        # Include protected parents' descendants. Do not treat Explorer's entire
        # launch tree as a protected family: it is the parent of most desktop apps.
        protected |= {pid for pid, p in processes.items()
                      if (p.get('name') or '').casefold() in
                      PROTECTED_NAMES - {'explorer.exe', 'dwm.exe'}}
        # Protect the current app's entire executable family, including renderers.
        fg_name = processes.get(foreground, {}).get('name')
        if not fg_name:
            raise OSError('Foreground process identity is unavailable')
        protected |= {pid for pid, p in processes.items() if p.get('name', '').casefold() == fg_name.casefold()}
        protected = self.expand_descendants(protected, processes)
        roots = [self.windows_root] + context.get('protected_roots', [])
        exes, paths = {}, {}
        def executable(pid):
            if pid not in paths:
                try:
                    paths[pid] = psutil.Process(pid).exe()
                except psutil.Error:
                    paths[pid] = None
            return paths[pid]
        for pid in windows:
            if pid in protected or processes.get(pid, {}).get('name', '').casefold() in PROTECTED_NAMES:
                continue
            exe = executable(pid)
            if exe:
                exes[path_key(exe)] = exe
        apps = []
        for key, exe in list(exes.items())[:128]:
            name = ntpath.basename(exe)
            if name.casefold() in PROTECTED_NAMES or any(within(exe, root) for root in roots):
                continue
            same_name = {pid for pid, p in processes.items() if p.get('name', '').casefold() == name.casefold()}
            # An unreadable same-name instance could be part of the application.
            if any(not executable(pid) for pid in same_name):
                continue
            same_exe = {pid for pid in same_name if path_key(executable(pid)) == key}
            members = self.expand_descendants(same_exe, processes)
            window_states = [v for pid in members for v in windows.get(pid, [])]
            reason = None
            if members & protected:
                reason = 'foreground_or_protected_family'
            elif not window_states or not all(window_states):
                reason = 'all_visible_windows_must_be_minimized'
            data, cpu, io, rss = [], 0.0, 0, 0
            for pid in sorted(members):
                child_exe = executable(pid)
                if (not child_exe or
                        ntpath.basename(child_exe).casefold() in PROTECTED_NAMES or
                        any(within(child_exe, root) for root in roots) or
                        not within(child_exe, ntpath.dirname(exe))):
                    reason = 'protected_or_unverified_family'
                    break
                try:
                    process = psutil.Process(pid)
                    if (self.session_id(pid) != self.session or
                            process.username().casefold() != self.username or
                            path_key(process.exe()) != path_key(child_exe)):
                        raise OSError('Process identity/session changed')
                    created = process.create_time()
                    times, counters, memory = process.cpu_times(), process.io_counters(), process.memory_info()
                    cpu += times.user + times.system
                    io += counters.read_bytes + counters.write_bytes + getattr(counters, 'other_bytes', 0)
                    rss += memory.rss
                    data.append({'pid': pid, 'created': created, 'exe': child_exe, 'rss': memory.rss})
                except (psutil.Error, OSError):
                    reason = 'process_telemetry_unavailable'
                    break
            if not data or len(data) != len(members):
                continue
            apps.append({'exe': exe, 'name': name, 'processes': data, 'rss': rss,
                         'cpu_seconds': cpu, 'io_bytes': io, 'safe': reason is None,
                         'reason': reason or 'observing_minimized_app'})
        return apps

    @staticmethod
    def available_ram():
        return psutil.virtual_memory().available / GIB

    def trim_one(self, identity, permitted):
        # No TERMINATE, SUSPEND or VM_WRITE access. Verify the opened handle, not
        # a separate PID lookup, so PID reuse cannot redirect this operation.
        handle = self.k32.OpenProcess(0x1000 | 0x0100, False, identity['pid'])
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            code = wintypes.DWORD()
            if (not self.k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]) or
                    not self.k32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) or
                    not self.k32.GetExitCodeProcess(handle, ctypes.byref(code))):
                raise ctypes.WinError(ctypes.get_last_error())
            created = ((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime) / 10000000 - 11644473600
            if (code.value != 259 or abs(created-identity['created']) > .01 or
                    path_key(buffer.value) != path_key(identity['exe'])):
                raise OSError('Opened process identity changed')
            before = psutil.Process(identity['pid']).memory_info()
            if not permitted() or self.foreground() in identity.get('family_pids', [identity['pid']]):
                return None
            if not self.psapi.EmptyWorkingSet(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                after = psutil.Process(identity['pid']).memory_info()
                after_rss = after.rss
            except psutil.Error:
                after_rss = None
            return {'pid': identity['pid'], 'before_rss_mib': before.rss/MIB,
                    'after_rss_mib': None if after_rss is None else after_rss/MIB,
                    'working_set_reduction_mib': None if after_rss is None else max(0, before.rss-after_rss)/MIB}
        finally:
            self.k32.CloseHandle(handle)


class AppMemoryReclaimer:
    def __init__(self, config, context, audit, backend=None, clock=time.monotonic):
        options = config.get('app_memory_reclaim', {})
        self.enabled = options.get('enabled', False) is True
        self.context, self.audit, self.backend, self.clock = context, audit, backend, clock
        self.tracker = IdleTracker()
        self.lock = threading.Lock()
        self.pressure_since = self.recovered_since = self.last_tick = None
        self.last_action = -math.inf
        self.cooldowns = {}
        self.episode_actions = 0
        self.recent = deque(maxlen=10)
        self.view = {'state': 'initializing' if self.enabled else 'disabled', 'apps': []}

    def status(self):
        with self.lock:
            return {'enabled': self.enabled, 'mode': 'minimized_idle_working_set_trim',
                    'keeps_processes_and_windows': True, 'releases_commit_or_vram': False,
                    'idle_seconds': IDLE_SECONDS, 'sustained_pressure_seconds': PRESSURE_SECONDS,
                    'per_app_cooldown_seconds': COOLDOWN_SECONDS, 'sample_seconds': INTERVAL,
                    'max_actions_per_pressure_episode': MAX_EPISODE_ACTIONS,
                    'limitations': 'idle is a conservative heuristic; pages can reload; RSS reduction is not guaranteed host gain',
                    **copy.deepcopy(self.view), 'recent_actions': copy.deepcopy(list(self.recent))}

    def publish(self, **values):
        with self.lock:
            self.view.update(values)

    def pressure(self, context):
        ram = self.backend.available_ram()
        target = context.get('required_ram_gib')
        valid = (context.get('allowed') is True and finite(ram) and ram >= 0
                 and finite(target) and target > 0)
        return bool(valid and ram < target), ram, target

    def tick(self):
        if not self.enabled:
            return
        if self.backend is None:
            self.backend = WindowsApps()
        now = self.clock()
        if self.last_tick is None or not 0 < now-self.last_tick <= INTERVAL*3:
            self.pressure_since = self.recovered_since = None
        self.last_tick = now
        context = self.context()
        apps = self.tracker.observe(self.backend.snapshot(context), now)
        self.cooldowns = {k: v for k, v in self.cooldowns.items() if now-v < COOLDOWN_SECONDS}
        pressured, ram, target = self.pressure(context)
        state = 'ram_sufficient' if context.get('allowed') else context.get('reason', 'no_model_task')
        if pressured:
            self.recovered_since = None
            if self.pressure_since is None:
                self.pressure_since = now
            state = 'waiting_for_sustained_ram_pressure'
        else:
            self.pressure_since = None
            if not context.get('allowed') or (finite(target) and finite(ram) and ram >= target+1):
                if self.recovered_since is None:
                    self.recovered_since = now
                if now-self.recovered_since >= 60:
                    self.episode_actions = 0
            else:
                self.recovered_since = None
        eligible = [a for a in apps if a['eligible'] and path_key(a['exe']) not in self.cooldowns]
        self.publish(last_scan=time.time(), available_ram_gib=ram, required_ram_gib=target,
                     model=context.get('model'), eligible_app_count=len(eligible), error=None,
                     apps=[{'name': a['name'], 'exe': a['exe'], 'rss_mib': round(a['rss']/MIB, 1),
                            'idle_seconds': round(a['idle_seconds'], 1), 'eligible': a in eligible,
                            'reason': ('cooldown' if path_key(a['exe']) in self.cooldowns else a['reason'])}
                           for a in sorted(apps, key=lambda a: a['rss'], reverse=True)[:20]])
        if pressured and now-self.pressure_since >= PRESSURE_SECONDS:
            if self.episode_actions >= MAX_EPISODE_ACTIONS:
                state = 'episode_limit_reached'
            elif now-self.last_action < ACTION_INTERVAL:
                state = 'waiting_for_action_interval'
            elif not eligible:
                state = 'no_verified_idle_apps'
            else:
                app = max(eligible, key=lambda a: a['rss'])
                key = path_key(app['exe'])
                before_available = ram
                results, errors = [], []

                def permitted():
                    current = self.context()
                    if not self.pressure(current)[0]:
                        return False
                    fresh = next((a for a in self.backend.snapshot(current) if path_key(a['exe']) == key), None)
                    return bool(fresh and fresh['safe'] and
                                quiet_delta(app, fresh, max(.001, self.clock()-now), immediate=True))

                # Recheck foreground, windows, identities, activity, task and RAM
                # before EACH process; partial success never implies whole-app gain.
                for process in sorted(app['processes'], key=lambda p: p['rss'], reverse=True)[:MAX_PROCESSES]:
                    try:
                        result = self.backend.trim_one(dict(process, family_pids=[p['pid'] for p in app['processes']]), permitted)
                        if result is None:
                            break
                        results.append(result)
                    except (psutil.Error, OSError):
                        errors.append({'pid': process['pid'], 'reason': 'process_changed_or_access_unavailable'})
                        break
                self.cooldowns[key] = now
                self.tracker.rows.pop(key, None)
                if results:
                    self.last_action = now
                    self.episode_actions += 1
                    event = {'time': time.time(), 'app': app['name'], 'exe': app['exe'],
                             'model': context.get('model'), 'required_ram_gib': target,
                             'before_available_ram_gib': before_available,
                             'after_available_ram_gib': self.backend.available_ram(),
                             'processes': results, 'errors': errors,
                             'effect': 'working_set_only; host change may include other activity'}
                    with self.lock:
                        self.recent.append(event)
                    self.audit('app_working_set_trimmed', event)
                    state = 'trimmed_waiting_for_normal_resource_measurement'
                else:
                    state = 'candidate_changed_or_access_unavailable'
        self.publish(state=state, episode_actions=self.episode_actions)

    def run(self, stop):
        while not stop.is_set():
            started = self.clock()
            try:
                self.tick()
            except Exception as exc:
                # A desktop/telemetry error must never impair the model monitor.
                self.tracker.rows.clear()
                self.pressure_since = self.recovered_since = None
                self.publish(state='telemetry_unavailable', error=type(exc).__name__, eligible_app_count=0, apps=[])
            stop.wait(max(.1, INTERVAL-(self.clock()-started)))
