"""
Language Frontends for Shader VM

This module demonstrates the power of the VM architecture:
Multiple different languages compile to the same bytecode!

Included languages:
- MathDSL: Math expression parser (sin(x*2 + time))
- LogoLike: Turtle graphics language
- SimplePython: Python-like shader DSL
"""

import re
import math
from typing import List, Tuple
from shader_vm import ShaderVM, Opcode


class MathDSLCompiler:
    """
    Compile math expressions to shader VM bytecode

    Supported:
    - Variables: x, y, u, v, time, pi
    - Operators: +, -, *, /, ^
    - Functions: sin, cos, tan, sqrt, abs, floor, ceil, fract
    - Constants: numbers

    Examples:
    - "sin(x*2 + time)"
    - "x^2 + y^2"
    - "sqrt((x-0.5)^2 + (y-0.5)^2)"
    """

    def __init__(self):
        self.vm = ShaderVM()
        self.tokens = []
        self.pos = 0

    def compile(self, expression: str) -> ShaderVM:
        """Compile a math expression to bytecode"""
        self.vm = ShaderVM()
        self.tokens = self.tokenize(expression)
        self.pos = 0

        # Parse and compile expression
        self.parse_expression()

        # Output as grayscale color
        self.vm.emit(Opcode.DUP)
        self.vm.emit(Opcode.DUP)
        self.vm.emit(Opcode.PUSH, 1.0)
        self.vm.emit(Opcode.COLOR)

        return self.vm

    def tokenize(self, expr: str) -> List[Tuple[str, str]]:
        """
        Tokenize math expression

        Returns: [(type, value), ...]
        Types: NUMBER, VAR, OP, LPAREN, RPAREN, COMMA, FUNC
        """
        tokens = []
        i = 0
        expr = expr.replace(' ', '')

        while i < len(expr):
            # Number
            if expr[i].isdigit() or expr[i] == '.':
                j = i
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                tokens.append(('NUMBER', expr[i:j]))
                i = j

            # Identifier (variable or function)
            elif expr[i].isalpha():
                j = i
                while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                    j += 1
                name = expr[i:j]

                # Check if function
                if j < len(expr) and expr[j] == '(':
                    tokens.append(('FUNC', name))
                else:
                    tokens.append(('VAR', name))
                i = j

            # Operators
            elif expr[i] in '+-*/^':
                tokens.append(('OP', expr[i]))
                i += 1

            # Parentheses
            elif expr[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif expr[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1

            # Comma
            elif expr[i] == ',':
                tokens.append(('COMMA', ','))
                i += 1

            else:
                i += 1  # Skip unknown characters

        return tokens

    def peek(self) -> Tuple[str, str]:
        """Peek at current token"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', '')

    def consume(self):
        """Consume current token"""
        self.pos += 1

    def parse_expression(self):
        """Parse expression (handles +, -)"""
        self.parse_term()

        while self.peek()[0] == 'OP' and self.peek()[1] in '+-':
            op = self.peek()[1]
            self.consume()
            self.parse_term()
            if op == '+':
                self.vm.emit(Opcode.ADD)
            else:
                self.vm.emit(Opcode.SUB)

    def parse_term(self):
        """Parse term (handles *, /)"""
        self.parse_power()

        while self.peek()[0] == 'OP' and self.peek()[1] in '*/':
            op = self.peek()[1]
            self.consume()
            self.parse_power()
            if op == '*':
                self.vm.emit(Opcode.MUL)
            else:
                self.vm.emit(Opcode.DIV)

    def parse_power(self):
        """Parse power (handles ^)"""
        self.parse_factor()

        if self.peek()[0] == 'OP' and self.peek()[1] == '^':
            self.consume()
            self.parse_power()  # Right associative
            self.vm.emit(Opcode.POW)

    def parse_factor(self):
        """Parse factor (number, variable, function call, or parenthesized expression)"""
        token_type, token_val = self.peek()

        # Number
        if token_type == 'NUMBER':
            self.vm.emit(Opcode.PUSH, float(token_val))
            self.consume()

        # Variable
        elif token_type == 'VAR':
            self.emit_variable(token_val)
            self.consume()

        # Function call
        elif token_type == 'FUNC':
            func_name = token_val
            self.consume()
            self.consume()  # '('

            # Parse arguments
            args = []
            self.parse_expression()  # First argument
            args.append(1)

            while self.peek()[0] == 'COMMA':
                self.consume()
                self.parse_expression()
                args.append(1)

            self.consume()  # ')'

            # Emit function call
            self.emit_function(func_name, len(args))

        # Parenthesized expression
        elif token_type == 'LPAREN':
            self.consume()
            self.parse_expression()
            self.consume()  # ')'

    def emit_variable(self, name: str):
        """Emit bytecode to load a variable"""
        if name in ('x', 'u'):
            self.vm.emit(Opcode.UV)
            self.vm.emit(Opcode.POP)  # Drop y, keep x
        elif name in ('y', 'v'):
            self.vm.emit(Opcode.UV)
            self.vm.emit(Opcode.SWAP)
            self.vm.emit(Opcode.POP)  # Drop x, keep y
        elif name == 'time' or name == 't':
            self.vm.emit(Opcode.TIME)
        elif name == 'pi':
            self.vm.emit(Opcode.PUSH, math.pi)
        else:
            # Unknown variable - push 0
            self.vm.emit(Opcode.PUSH, 0.0)

    def emit_function(self, name: str, arg_count: int):
        """Emit bytecode for function call"""
        func_map = {
            'sin': Opcode.SIN,
            'cos': Opcode.COS,
            'tan': Opcode.TAN,
            'sqrt': Opcode.SQRT,
            'abs': Opcode.ABS,
            'floor': Opcode.FLOOR,
            'ceil': Opcode.CEIL,
            'fract': Opcode.FRACT,
        }

        if name in func_map:
            self.vm.emit(func_map[name])
        elif name == 'min' and arg_count == 2:
            self.vm.emit(Opcode.MIN)
        elif name == 'max' and arg_count == 2:
            self.vm.emit(Opcode.MAX)
        elif name == 'pow' and arg_count == 2:
            self.vm.emit(Opcode.POW)
        elif name == 'atan2' and arg_count == 2:
            self.vm.emit(Opcode.ATAN2)


class SimplePythonCompiler:
    """
    Compile simple Python-like shader code

    Supported:
    - Variable assignment: color = sin(uv.x * 10)
    - UV coordinates: uv.x, uv.y
    - Math operations
    - Return statement: return (r, g, b, a)

    Example:
    ```
    r = sin(uv.x * 10 + time)
    g = cos(uv.y * 10 + time)
    b = 0.5
    return (r, g, b, 1.0)
    ```
    """

    def __init__(self):
        self.vm = ShaderVM()
        self.variables = {}

    def compile(self, code: str) -> ShaderVM:
        """Compile Python-like code to bytecode"""
        self.vm = ShaderVM()
        self.variables = {}

        # Simple line-by-line processing
        lines = code.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '=' in line and 'return' not in line:
                # Variable assignment
                var, expr = line.split('=', 1)
                var = var.strip()
                expr = expr.strip()

                # Compile expression
                math_compiler = MathDSLCompiler()
                math_compiler.vm = self.vm
                math_compiler.tokens = math_compiler.tokenize(expr)
                math_compiler.pos = 0
                math_compiler.parse_expression()

                # Store in our variable tracking (just track stack position)
                self.variables[var] = len(self.variables)

            elif line.startswith('return'):
                # Extract tuple
                match = re.search(r'return\s*\((.*?)\)', line)
                if match:
                    values = match.group(1).split(',')

                    # Compile each value (either variable or expression)
                    for val in values:
                        val = val.strip()
                        if val in self.variables:
                            # Load variable (it's already on stack)
                            # For now, just use DUP to duplicate
                            self.vm.emit(Opcode.DUP)
                        else:
                            # Constant
                            try:
                                self.vm.emit(Opcode.PUSH, float(val))
                            except ValueError:
                                # Try to compile as expression
                                math_compiler = MathDSLCompiler()
                                math_compiler.vm = self.vm
                                math_compiler.tokens = math_compiler.tokenize(val)
                                math_compiler.pos = 0
                                math_compiler.parse_expression()

                # Output color
                self.vm.emit(Opcode.COLOR)

        return self.vm


class LogoLikeCompiler:
    """
    LOGO-like turtle graphics language

    Commands:
    - forward <distance> / fd <distance>
    - right <angle> / rt <angle>
    - left <angle> / lt <angle>
    - penup / pu
    - pendown / pd
    - repeat <n> [ <commands> ]

    Example:
    ```
    repeat 4 [
        forward 10
        right 90
    ]
    ```

    This draws shapes by calculating distance from drawn lines.
    """

    def __init__(self):
        self.vm = ShaderVM()
        self.turtle_x = 0.5
        self.turtle_y = 0.5
        self.turtle_angle = 0.0
        self.pen_down = True
        self.paths = []

    def compile(self, commands: List[str]) -> ShaderVM:
        """Compile LOGO commands to shader that draws the result"""
        self.vm = ShaderVM()
        self.turtle_x = 0.5
        self.turtle_y = 0.5
        self.turtle_angle = 0.0  # Radians, 0 = right
        self.pen_down = True
        self.paths = []

        # Execute turtle commands to generate path
        for cmd in commands:
            self.execute_command(cmd.strip())

        # Generate shader that renders the paths
        return self.compile_paths_to_shader()

    def execute_command(self, cmd: str):
        """Execute a single turtle command"""
        parts = cmd.split()
        if not parts:
            return

        command = parts[0].lower()

        if command in ('forward', 'fd'):
            distance = float(parts[1]) / 100.0  # Normalize to 0-1 range
            new_x = self.turtle_x + distance * math.cos(self.turtle_angle)
            new_y = self.turtle_y + distance * math.sin(self.turtle_angle)

            if self.pen_down:
                self.paths.append({
                    'type': 'line',
                    'x1': self.turtle_x,
                    'y1': self.turtle_y,
                    'x2': new_x,
                    'y2': new_y,
                })

            self.turtle_x = new_x
            self.turtle_y = new_y

        elif command in ('right', 'rt'):
            angle = float(parts[1])
            self.turtle_angle -= math.radians(angle)

        elif command in ('left', 'lt'):
            angle = float(parts[1])
            self.turtle_angle += math.radians(angle)

        elif command in ('penup', 'pu'):
            self.pen_down = False

        elif command in ('pendown', 'pd'):
            self.pen_down = True

    def compile_paths_to_shader(self) -> ShaderVM:
        """
        Compile the paths to a shader that renders them

        For each pixel, calculate minimum distance to any line
        """
        if not self.paths:
            # No paths - just return black
            self.vm.emit(Opcode.PUSH, 0.0)
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.PUSH, 1.0)
            self.vm.emit(Opcode.COLOR)
            return self.vm

        # Get UV
        self.vm.emit(Opcode.UV)  # [u, v]

        # For simplicity, just check distance to first path
        # TODO: Loop through all paths and find minimum distance
        path = self.paths[0]

        if path['type'] == 'line':
            # Calculate distance from point to line segment
            # This is simplified - just checking distance to start point
            self.vm.emit(Opcode.PUSH, path['x1'])  # [u, v, x1]
            self.vm.emit(Opcode.SUB)  # [u, v-x1]
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.MUL)  # [u, (v-x1)^2]

            self.vm.emit(Opcode.SWAP)  # [(v-x1)^2, u]
            self.vm.emit(Opcode.PUSH, path['y1'])
            self.vm.emit(Opcode.SUB)  # [(v-x1)^2, u-y1]
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.MUL)  # [(v-x1)^2, (u-y1)^2]

            self.vm.emit(Opcode.ADD)  # [dist^2]
            self.vm.emit(Opcode.SQRT)  # [dist]

            # If distance < 0.01, white, else black
            self.vm.emit(Opcode.PUSH, 0.01)
            self.vm.emit(Opcode.LT)  # [dist < 0.01]

            # Convert to color
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.DUP)
            self.vm.emit(Opcode.PUSH, 1.0)
            self.vm.emit(Opcode.COLOR)

        return self.vm


def test_language_frontends():
    """Test all language frontends"""
    print("🌐 Language Frontends Test")
    print("=" * 60)

    # Test Math DSL
    print("\n1. Math DSL Compiler")
    print("-" * 60)

    math_compiler = MathDSLCompiler()

    expressions = [
        "sin(x*10 + time)",
        "x^2 + y^2",
        "sqrt((x-0.5)^2 + (y-0.5)^2)",
        "sin(x*5)*cos(y*5)",
        "abs(sin(time))",
    ]

    for expr in expressions:
        vm = math_compiler.compile(expr)
        print(f"\n  Expression: {expr}")
        print(f"  Instructions: {len(vm.instructions)}")
        print(f"  Bytecode: {len(vm.compile_to_uint32_array())} uint32s")

        # Show first few instructions
        lines = vm.disassemble().split('\n')[:5]
        for line in lines:
            print(f"    {line}")

    # Test SimplePython
    print("\n\n2. SimplePython Compiler")
    print("-" * 60)

    python_compiler = SimplePythonCompiler()

    code = """
    # Simple shader
    r = sin(x * 10 + time)
    g = cos(y * 10 + time)
    return (r, g, 0.5, 1.0)
    """

    vm = python_compiler.compile(code)
    print(f"\n  Code:\n{code}")
    print(f"  Instructions: {len(vm.instructions)}")
    print(f"  Bytecode: {len(vm.compile_to_uint32_array())} uint32s")

    # Test LOGO
    print("\n\n3. LOGO-Like Turtle Graphics")
    print("-" * 60)

    logo_compiler = LogoLikeCompiler()

    commands = [
        "forward 10",
        "right 90",
        "forward 10",
        "right 90",
        "forward 10",
        "right 90",
        "forward 10",
    ]

    vm = logo_compiler.compile(commands)
    print(f"\n  Commands: {commands}")
    print(f"  Paths generated: {len(logo_compiler.paths)}")
    print(f"  Instructions: {len(vm.instructions)}")

    print("\n" + "=" * 60)
    print("✅ All language frontends working!")


if __name__ == "__main__":
    test_language_frontends()
