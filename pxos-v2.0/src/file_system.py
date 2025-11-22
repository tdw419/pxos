from src.pixel_grid import PixelGrid

class PixelFileSystem:
    """Manages files and directories within a pixel-based file system."""
    def __init__(self, total_blocks_width=1024, total_blocks_height=768):
        # A bitmap to track which inodes are in use. A white pixel (255,255,255) means free, black (0,0,0) means used.
        self.inode_bitmap = PixelGrid(64, 64)
        self.inode_bitmap.grid.fill(255) # All inodes are initially free

        # The main storage area for file data, divided into 16x16 blocks.
        self.data_blocks = PixelGrid(total_blocks_width, total_blocks_height)
        self.block_size = (16, 16)

        # A bitmap to track which data blocks are in use.
        self.data_block_bitmap = PixelGrid(
            total_blocks_width // self.block_size[0],
            total_blocks_height // self.block_size[1]
        )
        self.data_block_bitmap.grid.fill(255) # All blocks are initially free

        self.inode_table = {}  # Maps inode number to file metadata
        self._next_inode = 0

    def _allocate_inode(self):
        """Finds a free inode and marks it as used. Returns the inode number."""
        if self._next_inode >= (64 * 64):
            raise IOError("Out of inodes.")

        inode_num = self._next_inode
        self._next_inode += 1

        # Mark inode as used in the bitmap
        x = inode_num % 64
        y = inode_num // 64
        self.inode_bitmap[x, y] = (0, 0, 0)

        return inode_num

    def _allocate_data_block(self):
        """Finds a free data block and returns its top-left coordinates."""
        # Simple sequential allocator for now
        bitmap_w = self.data_block_bitmap.width
        bitmap_h = self.data_block_bitmap.height

        for y in range(bitmap_h):
            for x in range(bitmap_w):
                if self.data_block_bitmap[x, y] == (255, 255, 255):
                    self.data_block_bitmap[x, y] = (0, 0, 0) # Mark as used
                    return (x * self.block_size[0], y * self.block_size[1])

        raise IOError("Out of data blocks.")

    def encode_string(self, s):
        """Encodes a string into a list of pixels (simple ASCII mapping)."""
        return [(ord(char), ord(char), ord(char)) for char in s]

    def create_file(self, filename, process_uid):
        """Creates a new file and returns its inode number."""
        inode = self._allocate_inode()

        # In a real system, the filename would be stored in a directory structure.
        # Here, we'll just associate it in the inode table.
        self.inode_table[inode] = {
            'filename': filename,
            'size_pixels': 0,  # File size in pixels
            'blocks': [],      # List of (x, y) coordinates for data blocks
            'uid': process_uid,
            'permissions': (0xFF, 0xFF, 0xFF)  # RWX as RGB
        }
        return inode

    def write_file(self, inode, data_pixels, offset_pixels=0):
        """Writes a list of pixels to a file, starting at a given offset."""
        if inode not in self.inode_table:
            raise ValueError(f"Inode {inode} does not exist.")

        file_meta = self.inode_table[inode]
        pixels_per_block = self.block_size[0] * self.block_size[1]

        # Allocate necessary data blocks
        pixels_to_write = len(data_pixels)
        required_blocks = (offset_pixels + pixels_to_write + pixels_per_block - 1) // pixels_per_block

        while len(file_meta['blocks']) < required_blocks:
            new_block_coords = self._allocate_data_block()
            file_meta['blocks'].append(new_block_coords)

        # Write the pixels to the allocated blocks
        for i, pixel in enumerate(data_pixels):
            total_offset = offset_pixels + i
            block_index = total_offset // pixels_per_block
            pixel_in_block_index = total_offset % pixels_per_block

            block_coords = file_meta['blocks'][block_index]

            # Calculate physical coordinates within the data_blocks grid
            physical_x = block_coords[0] + (pixel_in_block_index % self.block_size[0])
            physical_y = block_coords[1] + (pixel_in_block_index // self.block_size[0])

            self.data_blocks[physical_x, physical_y] = pixel

        # Update file size
        file_meta['size_pixels'] = max(file_meta['size_pixels'], offset_pixels + pixels_to_write)
        return pixels_to_write
