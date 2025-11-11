import os
import sys

# Add the root of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxos.compiler.transpiler import Transpiler

def test_hello_world_transpilation():
    """
    Tests that the hello.pxsl example transpiles correctly.
    """
    with open("examples/hello.pxsl", "r") as f:
        pxsl_code = f.read()

    transpiler = Transpiler(pxsl_code)
    wgsl_code = transpiler.transpile()

    print("--- Transpiled WGSL ---")
    print(wgsl_code)
    print("-----------------------")

    # Basic assertions to check for correctness
    assert "fn pixel_main_internal" in wgsl_code
    assert "let u = coord.x / screen_dims.x;" in wgsl_code
    assert "@compute @workgroup_size(8, 8, 1)" in wgsl_code

if __name__ == "__main__":
    test_hello_world_transpilation()
