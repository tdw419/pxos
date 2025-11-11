from pxos.compiler.parser import PxslParser
from pxos.compiler.templates import WGSL_TEMPLATE

class Transpiler:
    """
    Transpiles PxSL source code to WGSL source code.
    """

    def __init__(self, pxsl_source: str):
        self.pxsl_source = pxsl_source

    def transpile(self) -> str:
        """
        Performs the transpilation.
        """
        parser = PxslParser(self.pxsl_source)
        ast = parser.parse()

        # This is a simple substitution for now.
        # It will become more sophisticated as the parser develops.
        wgsl_code = WGSL_TEMPLATE.format(
            helper_functions="",  # Not implemented yet
            main_function_body=ast.main_function
        )

        return wgsl_code
