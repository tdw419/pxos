import struct
import collections
from collections import defaultdict
import copy
import os
import shutil
from pxos_filesystem import PxOSFilesystem

OP_HALT = 0x00
OP_IMM32 = 0x01
OP_ADD = 0x02
OP_CMP = 0x03
OP_JMP = 0x04
OP_JZ = 0x05
OP_JNZ = 0x06
OP_LOAD = 0x07
OP_STORE = 0x08
OP_FORK = 0x09
OP_SYSCALL = 0xF0

class ProcessState:
    def __init__(self, entry_point):
        self.ip = entry_point
        self.registers = [0] * 8
        self.halted = False

class PxVMScheduler:
    def __init__(self):
        self.processes = {}
        self.run_queue = collections.deque()
        self.sleep_queue = {}
        self.mailboxes = defaultdict(collections.deque)
        self.current_pid = None
        self.next_pid = 1
        self.tick_counter = 0

    def spawn(self, entry_point):
        pid = self.next_pid
        self.next_pid += 1
        state = ProcessState(entry_point)
        self.processes[pid] = state
        self.run_queue.append(pid)
        return pid

    def schedule(self):
        self.tick_counter += 1

        for pid, wake_tick in list(self.sleep_queue.items()):
            if self.tick_counter >= wake_tick:
                del self.sleep_queue[pid]
                self.run_queue.append(pid)

        if not self.run_queue:
            return None, None

        self.current_pid = self.run_queue.popleft()
        self.run_queue.append(self.current_pid)
        return self.current_pid, self.processes[self.current_pid]

    def yield_current_process(self):
        pass

    def sleep(self, ticks):
        pid = self.run_queue.pop()
        self.sleep_queue[pid] = self.tick_counter + ticks

    def exit(self):
        pid = self.run_queue.pop()
        del self.processes[pid]
        return self.run_queue[0] if self.run_queue else None

    def fork(self, parent_state):
        pid = self.next_pid
        self.next_pid += 1
        child_state = copy.deepcopy(parent_state)
        self.processes[pid] = child_state
        self.run_queue.append(pid)
        return pid

class PxVM:
    def __init__(self, imperfect=True):
        self.memory = bytearray()
        self.halted = False
        self.imperfect = imperfect
        self.host_terminal = None
        self.scheduler = PxVMScheduler()
        self.fs = PxOSFilesystem()
        self.current_process_state = None

        self.sysout: list[str] = []
        self.sys_messages = {
            1: "PXVM booting...",
            2: "PXVM ready.",
            3: "Task complete.",
            100: "pxVM v0.2 Terminal",
            101: "Welcome to pxVM -- the first self-writing kernel.",
            102: "pxOS> ",
            103: "Clock: ",
        }
        self.sys_colors = {
            1: (40, 40, 100, 255),
            2: (20, 20, 60, 255),
            3: (0, 0, 40, 255),
        }
        self.sys_layers = {
            1: "background",
            2: "ui",
            3: "vm",
            4: "overlay",
        }

    def load_program(self, program_words, entry_point):
        self.memory = bytearray(struct.pack(f'<{len(program_words)}I', *[val[0] for val in program_words]))
        self.scheduler.spawn(entry_point)

    def fetch_byte(self):
        byte = self.memory[self.current_process_state.ip]
        self.current_process_state.ip += 1
        return byte

    def fetch_u32(self):
        ip = self.current_process_state.ip
        val = struct.unpack("<I", self.memory[ip:ip+4])[0]
        self.current_process_state.ip += 4
        return val

    def run(self):
        while not self.halted:
            self.step()

    def step(self):
        pid, self.current_process_state = self.scheduler.schedule()
        if not self.current_process_state:
            self.halted = True
            return

        if self.current_process_state.halted or self.current_process_state.ip >= len(self.memory):
            self.current_process_state.halted = True
            return

        opcode = self.fetch_byte()

        if opcode == OP_HALT:
            self.current_process_state.halted = True
        elif opcode == OP_IMM32:
            reg = self.fetch_byte()
            val = self.fetch_u32()
            if reg < len(self.current_process_state.registers):
                self.current_process_state.registers[reg] = val
            elif self.imperfect:
                self.sysout.append(f"# WARNING: invalid register {reg} for IMM32")
        elif opcode == OP_ADD:
            reg1 = self.fetch_byte()
            reg2 = self.fetch_byte()
            self.current_process_state.registers[reg1] += self.current_process_state.registers[reg2]
        elif opcode == OP_CMP:
            reg1 = self.fetch_byte()
            reg2 = self.fetch_byte()
            self.current_process_state.registers[0] = self.current_process_state.registers[reg1] - self.current_process_state.registers[reg2]
        elif opcode == OP_JMP:
            addr = self.fetch_u32()
            self.current_process_state.ip = addr
        elif opcode == OP_JZ:
            addr = self.fetch_u32()
            if self.current_process_state.registers[0] == 0:
                self.current_process_state.ip = addr
        elif opcode == OP_JNZ:
            addr = self.fetch_u32()
            if self.current_process_state.registers[0] != 0:
                self.current_process_state.ip = addr
        elif opcode == OP_LOAD:
            reg = self.fetch_byte()
            addr = self.fetch_u32()
            val = struct.unpack("<I", self.memory[addr:addr+4])[0]
            self.current_process_state.registers[reg] = val
        elif opcode == OP_STORE:
            addr = self.fetch_u32()
            reg = self.fetch_byte()
            val = self.current_process_state.registers[reg]
            self.memory[addr:addr+4] = struct.pack("<I", val)
        elif opcode == OP_SYSCALL:
            num = self.fetch_byte()
            self.current_process_state.registers[0] = num
            try:
                self.handle_syscall(num)
            except Exception as e:
                if self.imperfect:
                    self.sysout.append(f"# ERROR: syscall {num} raised {e}")
                else:
                    raise
        elif self.imperfect:
            self.sysout.append(f"# WARNING: unknown opcode {opcode:02X}")

    def handle_syscall(self, num: int):
        r = self.current_process_state.registers

        if num == 1:  # SYS_PRINT_ID
            msg_id = r[1]
            msg = self.sys_messages.get(msg_id)
            if msg is None:
                self.sysout.append(f"PRINT [vm warn] unknown message_id {msg_id}")
            else:
                self.sysout.append(f"PRINT PXVM: {msg}")
        elif num == 2:  # SYS_RECT_ID
            x, y, w, h, color_id = r[1], r[2], r[3], r[4], r[5]
            rgba = self.sys_colors.get(color_id, (255, 0, 255, 255))
            if color_id not in self.sys_colors:
                self.sysout.append(f"# WARNING: unknown color_id {color_id}, using fallback")
            r_, g_, b_, a_ = rgba
            self.sysout.append(f"RECT {x} {y} {w} {h} {r_} {g_} {b_} {a_}")
        elif num == 3:  # SYS_TEXT_ID
            x, y, color_id, msg_id = r[1], r[2], r[3], r[4]
            rgba = self.sys_colors.get(color_id, (255, 255, 255, 255))
            msg = self.sys_messages.get(msg_id, f"[vm warn] unknown message_id {msg_id}")
            r_, g_, b_, a_ = rgba
            self.sysout.append(f"TEXT {x} {y} {r_} {g_} {b_} {a_} {msg}")
        elif num == 4:  # SYS_LAYER_USE_ID
            layer_id = r[1]
            name = self.sys_layers.get(layer_id)
            if name is None:
                self.sysout.append(f"PRINT [vm warn] unknown layer_id {layer_id}")
            else:
                self.sysout.append(f"LAYER USE {name}")
        elif num == 8: # SYS_POLL_EVENT
            if self.host_terminal and self.host_terminal.event_queue:
                ev_type, d1, d2, d3 = self.host_terminal.event_queue.popleft()
                r[0], r[1], r[2], r[3] = ev_type, d1, d2, d3
                self.sysout.append(f"# EVENT {ev_type} ({d1},{d2},{d3})")
            else:
                r[0] = 0 # EVENT_NONE
        elif num == 9: # SYS_DRAW_CHAR
            x, y, char_code, fg_id, bg_id = r[1], r[2], r[3], r[4], r[5]
            fg = self.sys_colors.get(fg_id, (255, 255, 255, 255))
            bg = self.sys_colors.get(bg_id, (0, 0, 0, 255))
            if self.host_terminal:
                self.host_terminal.draw_char_vm(x, y, char_code, fg, bg)
        elif num == 10: # SYS_DRAW_STRING
            x, y, msg_id, fg_id, bg_id = r[1], r[2], r[3], r[4], r[5]
            text = self.sys_messages.get(msg_id, "?")
            fg = self.sys_colors.get(fg_id, (255, 255, 255, 255))
            bg = self.sys_colors.get(bg_id, (0, 0, 0, 255))
            if self.host_terminal:
                for i, ch in enumerate(text):
                    self.host_terminal.draw_char_vm(x + i * 8, y, ord(ch), fg, bg)
        elif num == 11: # FORK
            child_pid = self.scheduler.fork(self.current_process_state)
            r[0] = child_pid
        elif num == 12: # YIELD
            self.scheduler.yield_current_process()
        elif num == 13: # SLEEP
            ticks = r[1]
            self.scheduler.sleep(ticks)
        elif num == 14: # IPC_SEND
            target_pid, msg_id = r[1], r[2]
            if target_pid not in self.scheduler.processes:
                self.sysout.append(f"# IPC WARN: PID {target_pid} does not exist")
                return
            self.scheduler.mailboxes[target_pid].append((self.scheduler.current_pid, msg_id))
            self.sysout.append(f"# IPC: {self.scheduler.current_pid} -> {target_pid}: msg {msg_id}")
        elif num == 15: # IPC_RECV
            if self.scheduler.mailboxes[self.scheduler.current_pid]:
                sender_pid, msg_id = self.scheduler.mailboxes[self.scheduler.current_pid].popleft()
                r[0], r[1] = msg_id, sender_pid
                self.sysout.append(f"# IPC: {self.scheduler.current_pid} <- {sender_pid}: msg {msg_id}")
            else:
                self.scheduler.yield_current_process()
                self.current_process_state.ip -= 2
        elif num == 16: # EXIT
            next_pid = self.scheduler.exit()
            if not next_pid:
                self.halted = True
        elif num == 20: # FS_OPEN
            filename_id, mode = r[1], r[2]
            r[0] = self.fs.open(filename_id, mode)
        elif num == 21: # FS_CLOSE
            handle = r[1]
            r[0] = self.fs.close(handle)
        elif num == 22: # FS_WRITE
            handle, buf_ptr, length = r[1], r[2], r[3]
            data = self.memory[buf_ptr:buf_ptr+length]
            r[0] = self.fs.write(handle, data)
        elif num == 23: # FS_READ
            handle, buf_ptr, length = r[1], r[2], r[3]
            data = self.fs.read(handle, length)
            self.memory[buf_ptr:buf_ptr+len(data)] = data
            r[0] = len(data)
        elif num == 99: # SYS_SELF_MODIFY
            candidate_id = r[1]
            force_rollback = r[2] if len(r) > 2 else 0
            candidate_name = self.fs.names.get(candidate_id, f"kernel_{candidate_id}.bin")
            candidate_path = f"./pxos_fs/{candidate_name}"
            current_path = "./pxos_fs/kernel_current.bin"
            backup_path = "./pxos_fs/kernel_backup.bin"
            validation_path = "./pxos_fs/kernel_validated.bin"

            if os.path.exists(current_path):
                shutil.copy2(current_path, backup_path)
                self.sysout.append("# SELF_MODIFY: Soul backed up")

            if force_rollback:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, current_path)
                    self.sysout.append("# SELF_MODIFY: Rollback complete — soul restored")
                    self.host_terminal.reboot_with_new_kernel(current_path)
                return

            if not os.path.exists(candidate_path):
                self.sysout.append(f"# SELF_MODIFY ERROR: {candidate_name} not found")
                self._restore_and_reboot(backup_path)
                return

            try:
                with open(candidate_path, "rb") as f:
                    data = f.read()
                    if len(data) < 16 or len(data) % 4 != 0:
                        raise ValueError("Invalid binary format")

                shutil.copy2(candidate_path, validation_path)
                shutil.copy2(validation_path, current_path)

                self.sysout.append(f"# SELF_MODIFY: New soul accepted — {candidate_name}")
                self.sysout.append("# SELF_MODIFY: Rebooting into evolved self...")
                self.host_terminal.print_line("Transcendence sequence initiated...")
                self.host_terminal.reboot_with_new_kernel(current_path)
            except Exception as e:
                self.sysout.append(f"# SELF_MODIFY FAILED: {e}")
                self.sysout.append("# SELF_MODIFY: Restoring last known good soul...")
                self._restore_and_reboot(backup_path)

        elif num == 30: # SYS_VERSION_GET
            pass # Not implemented
        elif num == 31: # SYS_VERSION_LIST
            pass # Not implemented
        elif num == 32: # SYS_VERSION_SWITCH
            pass # Not implemented
        elif num == 106: # SYS_PHEROMONE_DEPOSIT
            pass # Not implemented
        elif num == 107: # SYS_PHEROMONE_SNIFF
            pass # Not implemented
        elif num == 108: # SYS_MERGE_WITH
            pass # Not implemented
        elif num == 109: # SYS_SPECIATE
            pass # Not implemented
        else:
            self.sysout.append(f"# WARNING: unknown syscall {num} with args {list(r)}")

    def _restore_and_reboot(self, backup_path):
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, "./pxos_fs/kernel_current.bin")
            self.host_terminal.print_line("Soul restored from backup")
        else:
            self.host_terminal.print_line("No backup found — booting emergency kernel")
        self.host_terminal.reboot_with_new_kernel("./pxos_fs/kernel_current.bin")
