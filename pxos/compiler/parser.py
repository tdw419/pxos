import re
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PxslAst:
    """
    A simple Abstract Syntax Tree for a PxSL script.
    For now, it just separates the main function from helpers.
    """
    main_function: str = ""
    helper_functions: List[str] = field(default_factory=list)
    other_code: str = ""


class PxslParser:
    """
    A basic parser for the PxSL language.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code.strip()

    def parse(self) -> PxslAst:
        """
        Parses the source code and returns a PxslAst.
        """
        ast = PxslAst()

        # For simplicity, we'll use a regex to find function definitions
        # This is a naive implementation and will be improved later.
        function_pattern = re.compile(r"fn\s+(\w+)\s*\([^)]*\)\s*->\s*[\w<>]*\s*\{.*?}", re.DOTALL)

        main_func_match = re.search(r"fn\s+pixel_main\s*\([^)]*\)\s*->\s*[\w<>]*\s*\{(.*)\}", self.source_code, re.DOTALL)
        if main_func_match:
            ast.main_function = main_func_match.group(1).strip()
        else:
            # If no explicit pixel_main, assume the whole script is the body.
            ast.main_function = self.source_code

        return ast
