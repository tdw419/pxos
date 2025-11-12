import ast
import inspect
import textwrap
from pxos.gpu_types import GPUBuffer
from pxos.compiler.templates import WGSL_TEMPLATE

TYPE_MAP = {
    float: "f32",
    int: "i32",
    bool: "bool",
}

class PythonASTParser:
    """
    Parses a Python function into an Abstract Syntax Tree (AST).
    """
    def __init__(self, func):
        self.func = func
        self.source = inspect.getsource(func)
        self.source = textwrap.dedent(self.source)

    def parse(self) -> ast.Module:
        """
        Parses the function's source code and returns the AST.
        """
        return ast.parse(self.source)

class ASTToWGSLTransformer(ast.NodeVisitor):
    """
    Walks a Python AST and generates a WGSL code string.
    """
    def __init__(self, func_sig):
        self.func_sig = func_sig
        self.wgsl_code = []
        self.buffer_bindings = []
        self.declared_variables = set()
        self.indent_level = 0

    def transform(self, node: ast.AST) -> str:
        """
        Transforms the given AST node into a WGSL string.
        """
        self.wgsl_code = []
        self.buffer_bindings = []
        self.declared_variables = set()
        self.visit(node)

        bindings_str = "\n".join(self.buffer_bindings)
        body_str = "".join(self.wgsl_code)

        return WGSL_TEMPLATE.format(
            buffer_bindings=bindings_str,
            kernel_body=body_str
        )

    def _indent(self):
        return "    " * self.indent_level

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._generate_buffer_bindings(node)
        self.indent_level += 1
        # Add the global invocation id to the set of declared variables
        self.declared_variables.add("i")
        self.wgsl_code.append(f"{self._indent()}let i = global_id.x;\n")
        for statement in node.body:
            self.visit(statement)
        self.indent_level -= 1

    def _generate_buffer_bindings(self, node: ast.FunctionDef):
        binding_index = 0
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self.func_sig.parameters[param_name].annotation

            if "GPUBuffer" in str(param_type):
                element_type = param_type.__args__[0]
                wgsl_type = TYPE_MAP.get(element_type, "f32")

                binding = f"@group(0) @binding({binding_index}) var<storage, read_write> {param_name}: array<{wgsl_type}>;"
                self.buffer_bindings.append(binding)
                binding_index += 1

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise NotImplementedError("Only single assignment is supported.")

        target_node = node.targets[0]
        target = self.visit(target_node)
        value = self.visit(node.value)

        if isinstance(target_node, ast.Subscript):
            self.wgsl_code.append(f"{self._indent()}{target} = {value};\n")
        else:
            if target in self.declared_variables:
                self.wgsl_code.append(f"{self._indent()}{target} = {value};\n")
            else:
                self.declared_variables.add(target)
                self.wgsl_code.append(f"{self._indent()}var {target} = {value};\n")


    def visit_Subscript(self, node: ast.Subscript) -> str:
        value = self.visit(node.value)
        slice_val = self.visit(node.slice)
        return f"{value}[{slice_val}]"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = { ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', }
        op = op_map.get(type(node.op))
        if op is None:
            raise NotImplementedError(f"Unsupported binary operator: {type(node.op)}")
        return f"({left} {op} {right})"

    def visit_Name(self, node: ast.Name) -> str:
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, float):
            return f"{node.value}f"
        return str(node.value)

    def visit_Return(self, node: ast.Return):
        pass

    def generic_visit(self, node):
        super().generic_visit(node)

class GpuKernel:
    def __init__(self, func):
        self.func = func
        self.parser = PythonASTParser(func)
        self.func_sig = inspect.signature(func)
        self.transformer = ASTToWGSLTransformer(self.func_sig)
        self.wgsl_code = None

    def __call__(self, *args, **kwargs):
        if self.wgsl_code is None:
            self.compile()
        print(f"Executing kernel '{self.func.__name__}' (simulation)...")
        pass

    def compile(self):
        ast_tree = self.parser.parse()
        self.wgsl_code = self.transformer.transform(ast_tree)

def gpu_kernel(func):
    return GpuKernel(func)
