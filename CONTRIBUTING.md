# Contributing to pxOS/pxvm

Thank you for your interest in contributing! This guide will help you get started.

---

## 🌟 Ways to Contribute

- **Write new organisms** - Create interesting kernel programs
- **Add features** - Extend the VM with new syscalls or instructions
- **Fix bugs** - Report and fix issues
- **Improve documentation** - Clarify or expand existing docs
- **Add tests** - Improve test coverage
- **Optimize performance** - Speed improvements

---

## 📝 Writing New Organisms

### Simple Organism Template

Create a file in `pxvm/examples/your_organism.asm`:

```asm
# your_organism.asm - Brief description
# Demonstrates: what behavior this shows

    # Setup phase
    MOV R0, 500         # X position
    MOV R1, 500         # Y position
    MOV R2, 0xFF00FF    # Color

main_loop:
    # Draw self
    PLOT

    # Your organism's behavior here
    # ...

    # Loop
    JMP main_loop
```

### Running Your Organism

```python
from pxvm.vm import PxVM
from pxvm.assembler import assemble

with open('pxvm/examples/your_organism.asm') as f:
    code = assemble(f.read())

vm = PxVM()
vm.spawn_kernel(code, color=0xFF00FF)
vm.run(max_cycles=1000)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(vm.framebuffer)
plt.show()
```

---

## 🔧 Adding New Syscalls

To add a new syscall (e.g., `SYS_MY_FEATURE = 105`):

### 1. Define the Opcode

Edit `pxvm/vm.py`:

```python
class PxVM:
    # ... existing opcodes ...
    OP_SYS_MY_FEATURE = 105  # Brief description
```

### 2. Implement the Handler

In `pxvm/vm.py`, inside the `step()` method:

```python
elif opcode == self.OP_SYS_MY_FEATURE:
    # Read arguments from registers
    arg1 = kernel.regs[0]
    arg2 = kernel.regs[1]

    # Implement behavior
    result = your_implementation(arg1, arg2)

    # Write result back
    kernel.regs[0] = result
```

### 3. Add to Assembler

Edit `pxvm/assembler.py`:

```python
OPCODES = {
    # ... existing opcodes ...
    'SYS_MY_FEATURE': 105,
}
```

And in `_estimate_size()` and `_emit_instruction()`:

```python
elif op in ('SYS_EMIT_PHEROMONE', ..., 'SYS_MY_FEATURE'):
    return 1  # If register-only

# And in _emit_instruction:
elif op in ('SYS_EMIT_PHEROMONE', ..., 'SYS_MY_FEATURE'):
    self.code.append(self.OPCODES[op])
    self.pc += 1
```

### 4. Document It

Add to `pxvm/README.md` in the Syscalls section:

```markdown
| Opcode | Mnemonic | Args | Description |
|--------|----------|------|-------------|
| 105 | `SYS_MY_FEATURE` | R0=arg1, R1=arg2 → R0=result | What it does |
```

### 5. Create a Demo

Create `demo_my_feature.py`:

```python
from pxvm.vm import PxVM
from pxvm.assembler import assemble

code = assemble("""
    MOV R0, 100
    MOV R1, 200
    SYS_MY_FEATURE
    HALT
""")

vm = PxVM()
vm.spawn_kernel(code)
vm.run()
print("Result:", vm.kernels[0].regs[0])
```

---

## 🧪 Adding Tests

Create or update test files:

```python
# test_my_feature.py
from pxvm.vm import PxVM
from pxvm.assembler import assemble

def test_my_feature():
    code = assemble("""
        MOV R0, 10
        MOV R1, 20
        SYS_MY_FEATURE
        HALT
    """)

    vm = PxVM()
    vm.spawn_kernel(code)
    vm.run(max_cycles=10)

    assert vm.kernels[0].regs[0] == expected_value
    print("✓ test_my_feature passed")

if __name__ == '__main__':
    test_my_feature()
```

---

## 📚 Documentation Standards

### Code Comments

```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of what this does.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
    pass
```

### Assembly Comments

```asm
# Clear description of what this section does
    MOV R0, 100         # What this specific line does
    ADD R1, R2          # Be specific
```

---

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: What went wrong?
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen?
4. **Actual behavior**: What actually happened?
5. **Environment**: Python version, OS, dependencies

Example:

```markdown
**Bug**: Pheromone diffusion causes crash

**Reproduce**:
```python
vm = PxVM()
vm.pheromone[500, 500] = 1000  # Set value > 255
vm.step()  # Crashes here
```

**Expected**: Should clamp to 255
**Actual**: Raises ValueError
**Environment**: Python 3.11, scipy 1.16.3
```

---

## 🔀 Pull Request Process

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** and test thoroughly
4. **Commit** with clear messages:
   ```
   Add SYS_MY_FEATURE syscall

   - Implements feature X
   - Adds demo_my_feature.py
   - Updates documentation
   ```
5. **Push** to your fork
6. **Open a pull request** with:
   - Clear description of changes
   - Why this change is needed
   - Any breaking changes

---

## 🎨 Code Style

### Python

- Follow PEP 8
- Use type hints where helpful
- Keep functions focused and small
- Use descriptive variable names

```python
# Good
def spawn_child(parent: Kernel, x: int, y: int) -> int:
    """Create child kernel at position."""
    pass

# Avoid
def sc(p, x, y):
    pass
```

### Assembly

- Use lowercase for opcodes
- Align comments
- Use descriptive labels
- Add section headers

```asm
# Good
main_loop:
    MOV R0, 100         # X position
    MOV R1, 200         # Y position
    PLOT                # Draw pixel
    JMP main_loop

# Avoid
L1: MOV R0,100
PLOT
JMP L1
```

---

## 💡 Ideas for Contributions

### Easy

- Add more example organisms
- Improve error messages
- Add docstrings to functions
- Fix typos in documentation

### Medium

- Add visualization tools
- Implement new syscalls
- Optimize pheromone diffusion
- Add save/load VM state

### Advanced

- Implement mutation system
- Add energy/hunger mechanics
- Create evolution framework
- Build interactive GUI
- JIT compilation for speed

---

## 🤝 Community Guidelines

- Be respectful and constructive
- Help others learn
- Share interesting organisms you create
- Document your work
- Test before submitting

---

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Documentation**: Read pxvm/README.md first
- **Examples**: Check pxvm/examples/ directory

---

## 🎯 Current Priorities

See the project roadmap:

**High Priority:**
- [ ] Mutation system
- [ ] Energy/hunger mechanics
- [ ] Death/aging system

**Medium Priority:**
- [ ] More example organisms
- [ ] Performance optimization
- [ ] Better visualization tools

**Low Priority:**
- [ ] GUI interface
- [ ] Networking/multi-VM
- [ ] Persistence

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the digital biosphere!**

*Every organism, every feature, every bug fix helps build a richer world.*
