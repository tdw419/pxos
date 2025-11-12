WGSL_TEMPLATE = """
{buffer_bindings}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
{kernel_body}
}}
"""
