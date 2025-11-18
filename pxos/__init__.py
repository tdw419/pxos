"""
High-level Python API for pxOS primitives.

These functions do NOT run on pxOS; they compile into low-level primitives.
"""
def print_text(text: str, row: int, col: int):
    """
    Prints a string to the screen at the specified location.

    This function is a high-level abstraction that will be compiled into a
    series of `WRITE` primitives to a video memory address.
    """
    pass

def clear_screen():
    """
    Clears the entire screen.

    This function is a high-level abstraction that will be compiled into a
    loop that writes spaces to the entire video memory.
    """
    pass

def move_cursor(row: int, col: int):
    """
    Moves the cursor to the specified location.

    This function is a high-level abstraction that will be compiled into
    a BIOS interrupt call.
    """
    pass
