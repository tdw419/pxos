import asyncio
import wgpu
from wgpu.utils import get_default_device
from PIL import Image
import numpy as np

# -------------------------------------------------------------------
# 1. Load PXI image (your "program")
# -------------------------------------------------------------------
def load_pxi_as_rgba(path: str) -> np.ndarray:
    try:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        return arr  # shape: (h, w, 4)
    except FileNotFoundError:
        print(f"Error: The program file '{path}' was not found.")
        print("Please create a 'program.pxi.png' file.")
        exit(1)

# -------------------------------------------------------------------
# 2. Create GPU objects
# -------------------------------------------------------------------
async def main():
    pxi = load_pxi_as_rgba("program.pxi.png")
    height, width, _ = pxi.shape

    # Use the get_default_device utility
    device = get_default_device()

    # PXI texture (read-only)
    pxi_tex = device.create_texture(
        size=(width, height, 1),
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        dimension="2d",
        format="rgba8unorm",
        mip_level_count=1,
        sample_count=1,
    )
    device.queue.write_texture(
        {"texture": pxi_tex},
        pxi.tobytes(),
        {"bytes_per_row": width * 4, "rows_per_image": height},
        (width, height, 1),
    )
    pxi_view = pxi_tex.create_view()

    # DATA / FRAME texture (we'll write here)
    frame_tex = device.create_texture(
        size=(width, height, 1),
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        dimension="2d",
        format="rgba8unorm",
        mip_level_count=1,
        sample_count=1,
    )
    frame_view = frame_tex.create_view()

    # ----------------------------------------------------------------
    # 3. WGSL "interpreter" – 1 pass = 1 cycle over all instruction pixels
    # ----------------------------------------------------------------
    shader_src = """
    @group(0) @binding(0)
    var pxi_tex: texture_2d<f32>;

    @group(0) @binding(1)
    var frame_img: texture_storage_2d<rgba8unorm, write>;

    // For now: decode just a few opcodes from whitepaper
    // R = opcode, G = xarg, B = yarg, A = flags
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let size = textureDimensions(pxi_tex);
        if (gid.x >= size.x || gid.y >= size.y) {
            return;
        }

        let uv = vec2<i32>(i32(gid.x), i32(gid.y));
        let px = textureLoad(pxi_tex, uv, 0);
        // px is vec4<f32> in 0..1; convert to 0..255
        let opcode = u32(round(px.r * 255.0));
        let xarg   = u32(round(px.g * 255.0));
        let yarg   = u32(round(px.b * 255.0));
        let flags  = u32(round(px.a * 255.0));

        // OPCODES
        // 0x00 = NOP
        // 0x60 = DRAW (write pixel to frame)
        // for demo: DRAW writes the instruction’s own color into frame region
        if (opcode == 0x60u) {
            textureStore(frame_img, uv, px);
        } else if (opcode == 0xFFu) {
            // HALT – do nothing for now
        } else {
            // others: no-op
        }
    }
    """

    shader = device.create_shader_module(code=shader_src)

    bind_group_layout = device.create_bind_group_layout(entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "texture": {"sample_type": "float", "view_dimension": "2d"},
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "storage_texture": {
                "access": "write-only",
                "format": "rgba8unorm",
                "view_dimension": "2d",
            },
        },
    ])

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"},
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": pxi_view},
            {"binding": 1, "resource": frame_view},
        ],
    )

    # ----------------------------------------------------------------
    # 4. Dispatch over whole texture
    # ----------------------------------------------------------------
    command_encoder = device.create_command_encoder()
    cpass = command_encoder.begin_compute_pass()
    cpass.set_pipeline(compute_pipeline)
    cpass.set_bind_group(0, bind_group, [], 0, 999999)
    workgroups_x = (width + 7) // 8
    workgroups_y = (height + 7) // 8
    cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1)
    cpass.end()

    device.queue.submit([command_encoder.finish()])

    # ----------------------------------------------------------------
    # 5. Read back FRAME texture so we can inspect it as an image
    # ----------------------------------------------------------------
    out_buf = device.create_buffer(
        size=width * height * 4,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder2 = device.create_command_encoder()
    encoder2.copy_texture_to_buffer(
        {"texture": frame_tex},
        {"buffer": out_buf, "bytes_per_row": width * 4, "rows_per_image": height},
        (width, height, 1),
    )
    device.queue.submit([encoder2.finish()])

    # The map_async is still async
    await out_buf.map_async(wgpu.MapMode.READ)
    data = out_buf.read_mapped()
    img_arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
    out_buf.unmap()

    # save for inspection
    out_img = Image.fromarray(img_arr, mode="RGBA")
    out_img.save("frame.output.png")
    print("Wrote frame.output.png")

if __name__ == "__main__":
    asyncio.run(main())
