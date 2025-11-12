import os
import sys
import pytest

# Add the root of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxos.compiler.python_transpiler import gpu_kernel
from pxos.gpu_types import GPUBuffer

def test_simple_buffer_kernel():
    """
    Tests a simple kernel that operates on a GPUBuffer.
    """
    @gpu_kernel
    def my_kernel(data: GPUBuffer[float]):
        i = global_id.x
        data[i] = data[i] * 2.0

    my_kernel.compile()
    wgsl_code = my_kernel.wgsl_code
    normalized_wgsl = "".join(wgsl_code.split())

    assert "@group(0)@binding(0)var<storage,read_write>data:array<f32>;" in normalized_wgsl
    assert "@compute@workgroup_size(64,1,1)" in normalized_wgsl
    assert "fnmain(@builtin(global_invocation_id)global_id:vec3<u32>)" in normalized_wgsl
    assert "leti=global_id.x;" in normalized_wgsl
    assert "data[i]=(data[i]*2.0f);" in normalized_wgsl

def test_variable_reassignment():
    """
    Tests that variable reassignment is handled correctly.
    """
    @gpu_kernel
    def reassignment_kernel(data: GPUBuffer[float]):
        i = global_id.x
        temp = data[i]
        temp = temp + 1.0
        data[i] = temp

    reassignment_kernel.compile()
    wgsl_code = reassignment_kernel.wgsl_code

    print("\n--- Generated WGSL (Reassignment) ---")
    print(wgsl_code)
    print("-------------------------------------")

    normalized_wgsl = "".join(wgsl_code.split())

    # Check for 'var' on first assignment
    assert "vartemp=data[i];" in normalized_wgsl
    # Check for simple assignment on the second
    assert "temp=(temp+1.0f);" in normalized_wgsl
    assert "data[i]=temp;" in normalized_wgsl

if __name__ == "__main__":
    pytest.main([__file__])
