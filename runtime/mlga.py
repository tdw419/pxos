
import struct

class MLGA:
    """
    Memory Layout Generator Agent (MLGA).
    Deterministically calculates byte offsets and alignment for WGSL structs.
    """

    @staticmethod
    def align_to(offset, alignment):
        return (offset + alignment - 1) & ~(alignment - 1)

    @staticmethod
    def get_layout(struct_def):
        """
        Given a list of member types, returns the offsets and total size/alignment.
        Supported types: 'u32', 'i32', 'f32', 'vec2f', 'vec3f', 'vec4f'

        struct_def example:
        [
            {'name': 'time', 'type': 'f32'},
            {'name': 'resolution', 'type': 'vec2f'},
            {'name': 'color', 'type': 'vec3f'}
        ]
        """

        type_info = {
            'bool':  {'size': 4, 'align': 4}, # WGSL bool is 4 bytes in uniform/storage
            'u32':   {'size': 4, 'align': 4},
            'i32':   {'size': 4, 'align': 4},
            'f32':   {'size': 4, 'align': 4},
            'vec2f': {'size': 8, 'align': 8},
            'vec3f': {'size': 12, 'align': 16}, # Special case: vec3 is 16-byte aligned
            'vec4f': {'size': 16, 'align': 16},
        }

        current_offset = 0
        max_align = 0
        layout = {}

        for member in struct_def:
            m_type = member['type']
            m_name = member['name']

            if m_type not in type_info:
                raise ValueError(f"Unknown type: {m_type}")

            info = type_info[m_type]
            align = info['align']
            size = info['size']

            # Update max alignment for the struct
            max_align = max(max_align, align)

            # Align current offset
            current_offset = MLGA.align_to(current_offset, align)

            layout[m_name] = {
                'offset': current_offset,
                'type': m_type,
                'size': size
            }

            current_offset += size

        # Structure end padding
        total_size = MLGA.align_to(current_offset, max_align)

        return {
            'members': layout,
            'total_size': total_size,
            'alignment': max_align
        }

# Example usage/Test
if __name__ == "__main__":
    # Test case: float, vec3 (should force padding)
    test_struct = [
        {'name': 'time', 'type': 'f32'},      # offset 0, size 4
        {'name': 'color', 'type': 'vec3f'}    # offset 16 (aligned), size 12
    ]
    # Expected: time @ 0..4. Next aligned to 16. color @ 16..28. Total size align 16 -> 32.

    layout = MLGA.get_layout(test_struct)
    print("Layout:", layout)
    # Output should reflect offsets.

    # For a simple storage buffer of array<u32>, MLGA is trivial (offset = index * 4)
    # But this class is ready for uniform buffers.
