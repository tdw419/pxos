import struct

OP_HALT = 0x00
OP_IMM32 = 0x01
OP_SYSCALL = 0xF0

class PxVM:
    def __init__(self, bytecode: bytes, imperfect=True):
        self.bytecode = bytecode
        self.ip = 0
        self.registers = [0] * 8
        self.halted = False
        self.imperfect = imperfect

        self.sysout: list[str] = []
        self.sys_messages = {
            1: "PXVM booting...",
            2: "PXVM ready.",
            3: "Task complete.",
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

    def fetch_byte(self):
        byte = self.bytecode[self.ip]
        self.ip += 1
        return byte

    def fetch_u32(self):
        val = struct.unpack("<I", self.bytecode[self.ip:self.ip+4])[0]
        self.ip += 4
        return val

    def run(self):
        while not self.halted:
            self.step()

    def step(self):
        if self.halted or self.ip >= len(self.bytecode):
            self.halted = True
            return

        opcode = self.fetch_byte()

        if opcode == OP_HALT:
            self.halted = True
        elif opcode == OP_IMM32:
            reg = self.fetch_byte()
            val = self.fetch_u32()
            if reg < len(self.registers):
                self.registers[reg] = val
            elif self.imperfect:
                self.sysout.append(f"# WARNING: invalid register {reg} for IMM32")
        elif opcode == OP_SYSCALL:
            num = self.fetch_byte()
            self.registers[0] = num
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
        r = self.registers

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

        else:
            self.sysout.append(f"# WARNING: unknown syscall {num} with args {list(r)}")
