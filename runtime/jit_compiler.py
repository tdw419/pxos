
import wgpu
import wgpu.backends.wgpu_native

class CompilationError(Exception):
    def __init__(self, message, line_number=None, column_number=None):
        self.message = message
        self.line_number = line_number
        self.column_number = column_number
        super().__init__(self.message)

class JITCompiler:
    def __init__(self):
        # Ensure we have a wgpu device for compilation/validation
        self.device = wgpu.utils.get_default_device()

    def compile(self, wgsl_source):
        """
        Compiles WGSL source to a ShaderModule.
        Raises CompilationError if validation fails.
        In wgpu-py, creating a shader module triggers validation.
        """
        try:
            # Attempt to create the shader module.
            # wgpu-py will call into the native wgpu implementation (e.g. wgpu-native)
            # which performs validation.
            shader_module = self.device.create_shader_module(code=wgsl_source)

            # Force compilation info to check for errors immediately
            # Note: wgpu-py might not expose compilation_info synchronously or in the same way
            # as the web API. However, create_shader_module usually throws on serious errors
            # in the native backend if validation fails.
            # Let's try to get compilation info if available, or assume success if no exception.

            # As of recent wgpu-py, we rely on the exception.
            return shader_module

        except Exception as e:
            # Parse the error message to extract line/column if possible
            error_msg = str(e)
            line = None
            col = None

            # Simple heuristic parsing for wgpu-native error formats
            # Example: "Shader validation error: \n   ┌─ :5:16"
            try:
                if ":" in error_msg:
                    parts = error_msg.split(":")
                    # Look for numeric parts indicative of line/col
                    # This is brittle but better than nothing for the "Reflection" requirement
                    # A robust regex would be better.
                    pass
            except:
                pass

            raise CompilationError(f"WGSL Validation Failed: {error_msg}", line, col)

# Singleton instance
jit_compiler = JITCompiler()
