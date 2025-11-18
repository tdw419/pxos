"""
Python to pxIR Frontend

Compiles a restricted subset of Python to pxIR.

Supported Python subset:
- Function definitions
- Basic arithmetic (+, -, *, /, @)
- Function calls
- Control flow (if/else, return)
- NumPy-style operations (for matrix ops)

Example:
    def forward(h, W, b):
        logits = h @ W + b
        return relu(logits)

Compiles to pxIR with MATMUL, ADD, RELU operations.
"""

import ast
from typing import Dict, Optional, List
from pathlib import Path

from .ir import (
    IRBuilder, Type, Value, Program, Block
)


class PythonToPxIR(ast.NodeVisitor):
    """Compile Python AST to pxIR"""

    def __init__(self):
        self.builder = IRBuilder()
        self.current_function: Optional[str] = None
        self.values: Dict[str, Value] = {}  # Python variables → SSA values

    def compile(self, source: str, function_name: str = "main") -> Program:
        """Compile Python source to pxIR"""
        tree = ast.parse(source)

        # Find target function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                self._compile_function(node)
                break
        else:
            raise ValueError(f"Function {function_name} not found")

        return self.builder.program

    def compile_file(self, source_file: Path, function_name: str = "main") -> Program:
        """Compile Python file to pxIR"""
        source = source_file.read_text()
        return self.compile(source, function_name)

    def _compile_function(self, node: ast.FunctionDef):
        """Compile function definition"""
        self.current_function = node.name

        # Create entry block
        entry = self.builder.create_block(f"{node.name}_entry")
        self.builder.set_insert_point(entry)

        # Add function parameters as values
        for arg in node.args.args:
            # For now, assume all params are f32
            # TODO: Infer types from annotations
            param_type = Type.f32()
            param_value = Value(arg.arg, param_type)
            self.values[arg.arg] = param_value

        # Compile function body
        for stmt in node.body:
            self.visit(stmt)

        # Add implicit return if needed
        if not entry.terminator:
            self.builder.ret()

    def visit_Return(self, node: ast.Return):
        """Handle return statement"""
        if node.value:
            value = self._compile_expr(node.value)
            self.builder.ret(value)
        else:
            self.builder.ret()

    def visit_Assign(self, node: ast.Assign):
        """Handle assignment"""
        # Compile RHS
        value = self._compile_expr(node.value)

        # Assign to LHS (only simple names for now)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.values[target.id] = value
            else:
                raise NotImplementedError(f"Assignment to {type(target)} not supported")

    def visit_Expr(self, node: ast.Expr):
        """Handle expression statement"""
        self._compile_expr(node.value)

    def _compile_expr(self, node: ast.expr) -> Value:
        """Compile expression to SSA value"""
        if isinstance(node, ast.BinOp):
            return self._compile_binop(node)
        elif isinstance(node, ast.Name):
            return self._compile_name(node)
        elif isinstance(node, ast.Call):
            return self._compile_call(node)
        elif isinstance(node, ast.Constant):
            return self._compile_constant(node)
        else:
            raise NotImplementedError(f"Expression {type(node)} not supported")

    def _compile_binop(self, node: ast.BinOp) -> Value:
        """Compile binary operation"""
        lhs = self._compile_expr(node.left)
        rhs = self._compile_expr(node.right)

        if isinstance(node.op, ast.Add):
            return self.builder.add(lhs, rhs)
        elif isinstance(node.op, ast.Sub):
            return self.builder.emit_op("SUB", [lhs, rhs], result_type=lhs.type)
        elif isinstance(node.op, ast.Mult):
            return self.builder.mul(lhs, rhs)
        elif isinstance(node.op, ast.Div):
            return self.builder.emit_op("DIV", [lhs, rhs], result_type=lhs.type)
        elif isinstance(node.op, ast.MatMult):
            # Matrix multiply (@)
            # TODO: Infer result type from lhs/rhs types
            result_type = lhs.type  # Placeholder
            return self.builder.matmul(lhs, rhs, result_type)
        else:
            raise NotImplementedError(f"Binary op {type(node.op)} not supported")

    def _compile_name(self, node: ast.Name) -> Value:
        """Compile variable reference"""
        if node.id in self.values:
            return self.values[node.id]
        else:
            raise ValueError(f"Undefined variable: {node.id}")

    def _compile_call(self, node: ast.Call) -> Value:
        """Compile function call"""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        else:
            raise NotImplementedError("Only simple function calls supported")

        # Compile arguments
        args = [self._compile_expr(arg) for arg in node.args]

        # Handle built-in/known functions
        if func_name == "relu":
            if len(args) != 1:
                raise ValueError("relu() takes exactly 1 argument")
            return self.builder.relu(args[0])

        elif func_name == "print":
            # Handle print() - convert to PRINT_STR
            if args and isinstance(node.args[0], ast.Constant):
                # String literal
                text = node.args[0].value
                # For now, emit as metadata
                self.builder.emit_op("PRINT_STR", [text], has_side_effects=True)
                # Return dummy value
                return self.builder.fresh_value(Type.i32())
            else:
                raise NotImplementedError("Only string literals in print() supported")

        else:
            # Generic function call
            # Assume result type is f32 for now
            result_type = Type.f32()
            return self.builder.emit_op("CALL", [func_name] + args, result_type=result_type)

    def _compile_constant(self, node: ast.Constant) -> Value:
        """Compile constant value"""
        value = node.value

        if isinstance(value, int):
            # Create constant as immediate
            # For now, just create a fresh value with constant metadata
            const_val = self.builder.fresh_value(Type.i32(), prefix="const")
            # Store constant in metadata
            const_val.metadata = {"constant": value}
            return const_val

        elif isinstance(value, float):
            const_val = self.builder.fresh_value(Type.f32(), prefix="const")
            const_val.metadata = {"constant": value}
            return const_val

        elif isinstance(value, str):
            # String constant
            const_val = self.builder.fresh_value(Type.ptr("mem", Type.i8()), prefix="str")
            const_val.metadata = {"constant": value}
            return const_val

        else:
            raise NotImplementedError(f"Constant type {type(value)} not supported")


# ============================================================================
# Convenience Functions
# ============================================================================

def compile_python(source: str, function: str = "main") -> Program:
    """Compile Python source to pxIR"""
    compiler = PythonToPxIR()
    return compiler.compile(source, function)


def compile_python_file(path: Path, function: str = "main") -> Program:
    """Compile Python file to pxIR"""
    compiler = PythonToPxIR()
    return compiler.compile_file(path, function)
