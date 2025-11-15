// frozen_display.wgsl

@group(0) @binding(0)
var samp : sampler;

@group(0) @binding(1)
var img : texture_2d<f32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VSOut {
    var out : VSOut;
    let x = f32(vertex_index / 2u) * 2.0 - 1.0;
    let y = f32(vertex_index % 2u) * 2.0 - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(
        (out.pos.x + 1.0) * 0.5,
        (1.0 - out.pos.y) * 0.5
    );
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    return textureSample(img, samp, in.uv);
}
