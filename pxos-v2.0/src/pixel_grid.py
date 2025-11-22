import numpy as np

class PixelGrid:
    """Represents a 2D grid of pixels."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Initialize a 3D numpy array for width, height, and color channels (RGB)
        self.grid = np.zeros((height, width, 3), dtype=np.uint8)

    def __getitem__(self, key):
        """Allows accessing pixel data using grid[x, y]."""
        x, y = key
        return tuple(self.grid[y, x])

    def __setitem__(self, key, value):
        """Allows setting pixel data using grid[x, y] = (r, g, b)."""
        x, y = key
        self.grid[y, x] = value

    def get_region(self, x, y, width, height):
        """Extracts a rectangular region from the grid."""
        return self.grid[y:y+height, x:x+width]

    def set_region(self, x, y, region_data):
        """Sets a rectangular region in the grid."""
        region_height, region_width, _ = region_data.shape
        self.grid[y:y+region_height, x:x+region_width] = region_data

    def to_image(self, library='PIL'):
        """Converts the pixel grid to an image object for saving or display."""
        if library == 'PIL':
            try:
                from PIL import Image
                return Image.fromarray(self.grid, 'RGB')
            except ImportError:
                raise ImportError("Pillow library is required to create an image. Please install it using 'pip install Pillow'")
        else:
            raise NotImplementedError(f"Image conversion with '{library}' is not supported.")

    def save(self, file_path):
        """Saves the pixel grid as a PNG image."""
        image = self.to_image()
        image.save(file_path)

    @classmethod
    def from_image(cls, file_path):
        """Loads a pixel grid from an image file."""
        try:
            from PIL import Image
            image = Image.open(file_path).convert('RGB')
            grid = cls(image.width, image.height)
            grid.grid = np.array(image)
            return grid
        except ImportError:
            raise ImportError("Pillow library is required to load an image. Please install it using 'pip install Pillow'")
