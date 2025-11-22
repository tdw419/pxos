
import numpy as np
from PIL import Image

def run_cpu_interpreter():
    """
    Simulates the Pixel VM on the CPU.
    """
    # 1. Load the program from the PNG file
    try:
        program_image = Image.open("programs/jump_test.png").convert("RGBA")
        program_pixels = program_image.load()
        program_width = program_image.width
    except FileNotFoundError:
        print("Error: programs/simple.png not found.")
        return

    # 2. Initialize the VM state
    ip = 0
    zero_flag = 0
    halted = False
    regs = np.zeros(16, dtype=np.uint32)
    data = np.array([5, 7, 0, 0, 0, 0, 0, 0], dtype=np.uint32)

    print("--- Starting CPU Interpreter ---")
    print(f"Initial Data: {data}")

    # 3. Execution loop
    max_steps = 100 # Safety break to prevent infinite loops
    for step in range(max_steps):
        if halted or ip >= program_width:
            break

        # Fetch and decode instruction
        r, g, b, a = program_pixels[ip, 0]
        opcode, arg0, arg1 = r, g, b

        next_ip = ip + 1

        # Execute instruction
        if opcode == 1: # LOAD
            regs[arg1] = data[arg0]
            print(f"Step {step}: LOAD data[{arg0}] ({data[arg0]}) -> reg[{arg1}]")
        elif opcode == 2: # STORE
            data[arg1] = regs[arg0]
            print(f"Step {step}: STORE reg[{arg0}] ({regs[arg0]}) -> data[{arg1}]")
        elif opcode == 3: # ADD
            print(f"Step {step}: ADD reg[{arg0}] ({regs[arg0]}) + reg[{arg1}] ({regs[arg1]}) -> reg[{arg1}]")
            regs[arg1] = np.uint32(regs[arg1] + regs[arg0])
        elif opcode == 4: # JUMP
            offset = np.uint8(arg0).astype(np.int8) # Interpret as signed 8-bit
            next_ip = np.uint32(np.int32(ip) + offset)
            print(f"Step {step}: JUMP by {offset} to {next_ip}")
        elif opcode == 5: # CMP
            zero_flag = 1 if regs[arg0] == regs[arg1] else 0
            print(f"Step {step}: CMP reg[{arg0}] ({regs[arg0]}) == reg[{arg1}] ({regs[arg1]}). zero_flag={zero_flag}")
        elif opcode == 6: # JNE
            if zero_flag == 0:
                offset = np.uint8(arg0).astype(np.int8)
                next_ip = np.uint32(np.int32(ip) + offset)
                print(f"Step {step}: JNE by {offset} to {next_ip} (condition met)")
            else:
                print(f"Step {step}: JNE (condition not met)")
        elif opcode == 255: # HALT
            halted = True
            print(f"Step {step}: HALT")
        else: # Unknown opcode
            halted = True
            print(f"Step {step}: UNKNOWN OPCODE ({opcode}), HALTING")

        ip = next_ip

    # 4. Print final results
    print("\n--- Execution Complete ---")
    print(f"Halted: {'Yes' if halted else 'No'}")
    print(f"Final IP: {ip}")
    print(f"Final Data: {data}")
    print(f"Final Registers: {regs}")

if __name__ == "__main__":
    run_cpu_interpreter()
