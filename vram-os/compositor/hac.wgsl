// hac.wgsl - Hardware Abstraction Compositor
// GPU-side window compositor for VRAM OS
// Reads the Window Table and composites all visible windows to the display buffer

// ============================================================================
// CONSTANTS
// ============================================================================

const WINDOW_TABLE_BASE: u32 = 0x100000u;  // 1MB offset in VRAM
const MAX_WINDOWS: u32 = 32u;
const ENTRY_SIZE: u32 = 16u;               // 16 bytes per entry
const DISPLAY_WIDTH: u32 = 1920u;
const DISPLAY_HEIGHT: u32 = 1080u;
const TEXTURE_WIDTH: u32 = 2048u;          // Power of 2 for efficient addressing

// ============================================================================
// STRUCTS
// ============================================================================

struct WindowEntry {
    // Pixel 0
    window_id: u32,
    x_pos: u32,
    y_pos: u32,
    flags: u32,

    // Pixel 1
    width: u32,        // In 8-pixel units
    height: u32,       // In 8-pixel units
    z_order: u32,
    state: u32,

    // Pixel 2
    title_ptr_high: u32,
    title_ptr_low: u32,
    title_length: u32,
    title_encoding: u32,

    // Pixel 3
    content_ptr_high: u32,
    content_ptr_low: u32,
    owner_pid: u32,
    permissions: u32,
};

struct CompositorParams {
    frame_count: u32,
    dirty_rect_count: u32,
    clear_color: u32,
    _padding: u32,
};

// ============================================================================
// BINDINGS
// ============================================================================

// VRAM texture: stores Window Table, program memory, and all data
@group(0) @binding(0) var vram_texture: texture_storage_2d<rgba8uint, read>;

// Display buffer: output framebuffer
@group(0) @binding(1) var display_buffer: texture_storage_2d<rgba8unorm, write>;

// Compositor parameters
@group(0) @binding(2) var<uniform> params: CompositorParams;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Convert linear byte address to 2D texture coordinates
fn addr_to_coord(addr: u32) -> vec2<i32> {
    let pixel_index = addr / 4u;  // 4 bytes per pixel (RGBA)
    let x = pixel_index % TEXTURE_WIDTH;
    let y = pixel_index / TEXTURE_WIDTH;
    return vec2<i32>(i32(x), i32(y));
}

// Load a 32-bit word from VRAM
fn load_word(addr: u32) -> u32 {
    let coord = addr_to_coord(addr);
    let pixel = textureLoad(vram_texture, coord);

    // Reconstruct 32-bit word from RGBA bytes
    return (u32(pixel.r) << 24u) |
           (u32(pixel.g) << 16u) |
           (u32(pixel.b) << 8u)  |
           u32(pixel.a);
}

// Load a single byte from VRAM
fn load_byte(addr: u32) -> u32 {
    let coord = addr_to_coord(addr & ~3u);  // Align to 4-byte boundary
    let pixel = textureLoad(vram_texture, coord);

    let byte_offset = addr & 3u;
    switch byte_offset {
        case 0u: { return u32(pixel.r); }
        case 1u: { return u32(pixel.g); }
        case 2u: { return u32(pixel.b); }
        case 3u: { return u32(pixel.a); }
        default: { return 0u; }
    }
}

// Load Window Entry from Window Table
fn load_window_entry(index: u32) -> WindowEntry {
    var entry: WindowEntry;

    let base_addr = WINDOW_TABLE_BASE + (index * ENTRY_SIZE);

    // Pixel 0
    entry.window_id = load_byte(base_addr + 0u);
    entry.x_pos = load_byte(base_addr + 1u);
    entry.y_pos = load_byte(base_addr + 2u);
    entry.flags = load_byte(base_addr + 3u);

    // Pixel 1
    entry.width = load_byte(base_addr + 4u);
    entry.height = load_byte(base_addr + 5u);
    entry.z_order = load_byte(base_addr + 6u);
    entry.state = load_byte(base_addr + 7u);

    // Pixel 2
    entry.title_ptr_high = load_byte(base_addr + 8u);
    entry.title_ptr_low = load_byte(base_addr + 9u);
    entry.title_length = load_byte(base_addr + 10u);
    entry.title_encoding = load_byte(base_addr + 11u);

    // Pixel 3
    entry.content_ptr_high = load_byte(base_addr + 12u);
    entry.content_ptr_low = load_byte(base_addr + 13u);
    entry.owner_pid = load_byte(base_addr + 14u);
    entry.permissions = load_byte(base_addr + 15u);

    return entry;
}

// Check if window is visible
fn is_visible(entry: WindowEntry) -> bool {
    // State 0xFF = free/unused
    if entry.state == 0xFFu {
        return false;
    }

    // Flags bit 0 = visible
    if (entry.flags & 0x01u) == 0u {
        return false;
    }

    return true;
}

// Get content buffer address for a window
fn get_content_address(entry: WindowEntry) -> u32 {
    return (entry.content_ptr_high << 8u) | entry.content_ptr_low;
}

// Load pixel from window's content buffer
fn load_window_pixel(entry: WindowEntry, x: u32, y: u32) -> vec4<f32> {
    let content_addr = get_content_address(entry);

    // Calculate pixel offset within window
    let pixel_offset = (y * entry.width * 8u + x) * 4u;  // 4 bytes per pixel
    let pixel_addr = content_addr + pixel_offset;

    let coord = addr_to_coord(pixel_addr);
    let pixel = textureLoad(vram_texture, coord);

    // Convert u8 to f32 normalized [0, 1]
    return vec4<f32>(
        f32(pixel.r) / 255.0,
        f32(pixel.g) / 255.0,
        f32(pixel.b) / 255.0,
        f32(pixel.a) / 255.0
    );
}

// Draw window decorations (title bar and border)
fn draw_decorations(entry: WindowEntry, screen_x: u32, screen_y: u32,
                   local_x: u32, local_y: u32) -> vec4<f32> {
    let decorated = (entry.flags & 0x04u) != 0u;

    if !decorated {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);  // No decorations
    }

    let width_px = entry.width * 8u;
    let height_px = entry.height * 8u;
    let title_bar_height = 24u;
    let border_width = 2u;

    // Title bar (top 24 pixels)
    if local_y < title_bar_height {
        let focused = (entry.flags & 0x02u) != 0u;
        if focused {
            return vec4<f32>(0.2, 0.4, 0.8, 1.0);  // Blue for focused
        } else {
            return vec4<f32>(0.5, 0.5, 0.5, 1.0);  // Gray for unfocused
        }
    }

    // Borders (2 pixels wide)
    if local_x < border_width || local_x >= width_px - border_width ||
       local_y < border_width || local_y >= height_px - border_width {
        return vec4<f32>(0.3, 0.3, 0.3, 1.0);  // Dark gray border
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);  // Transparent (use content)
}

// Alpha blend two colors
fn alpha_blend(src: vec4<f32>, dst: vec4<f32>) -> vec4<f32> {
    let src_alpha = src.a;
    let dst_alpha = dst.a;

    let out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha);

    if out_alpha < 0.001 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let out_rgb = (src.rgb * src_alpha + dst.rgb * dst_alpha * (1.0 - src_alpha)) / out_alpha;

    return vec4<f32>(out_rgb, out_alpha);
}

// ============================================================================
// COMPOSITOR MAIN
// ============================================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let screen_x = global_id.x;
    let screen_y = global_id.y;

    // Skip if outside display bounds
    if screen_x >= DISPLAY_WIDTH || screen_y >= DISPLAY_HEIGHT {
        return;
    }

    // Start with clear color (desktop background)
    var final_color = vec4<f32>(0.0, 0.2, 0.4, 1.0);  // Dark blue

    // Sort windows by Z-order (simplified: just iterate in order)
    // In a real implementation, we'd pre-sort or use a Z-buffer
    var z_sorted: array<u32, 32>;
    var valid_count = 0u;

    for (var i = 0u; i < MAX_WINDOWS; i = i + 1u) {
        let entry = load_window_entry(i);

        if is_visible(entry) {
            // Insert into sorted array
            var insert_pos = valid_count;
            for (var j = 0u; j < valid_count; j = j + 1u) {
                let other_entry = load_window_entry(z_sorted[j]);
                if entry.z_order < other_entry.z_order {
                    insert_pos = j;
                    break;
                }
            }

            // Shift elements right
            for (var k = valid_count; k > insert_pos; k = k - 1u) {
                z_sorted[k] = z_sorted[k - 1u];
            }

            z_sorted[insert_pos] = i;
            valid_count = valid_count + 1u;
        }
    }

    // Composite windows from back to front
    for (var i = 0u; i < valid_count; i = i + 1u) {
        let entry = load_window_entry(z_sorted[i]);

        let win_x = entry.x_pos;
        let win_y = entry.y_pos;
        let win_width = entry.width * 8u;
        let win_height = entry.height * 8u;

        // Check if this pixel is inside the window
        if screen_x >= win_x && screen_x < win_x + win_width &&
           screen_y >= win_y && screen_y < win_y + win_height {

            let local_x = screen_x - win_x;
            let local_y = screen_y - win_y;

            // Check for decorations
            let deco_color = draw_decorations(entry, screen_x, screen_y, local_x, local_y);

            var window_color: vec4<f32>;
            if deco_color.a > 0.5 {
                window_color = deco_color;
            } else {
                // Load pixel from window content
                window_color = load_window_pixel(entry, local_x, local_y);
            }

            // Apply transparency if enabled
            let transparent = (entry.flags & 0x40u) != 0u;
            if transparent {
                final_color = alpha_blend(window_color, final_color);
            } else {
                if window_color.a > 0.5 {
                    final_color = window_color;
                }
            }
        }
    }

    // Write to display buffer
    let display_coord = vec2<i32>(i32(screen_x), i32(screen_y));
    textureStore(display_buffer, display_coord, final_color);
}
