# The God Pixel and the Oracle Protocol

This document outlines the architecture of the God Pixel platform and the protocol for communication between pxOS organisms and the `SYS_LLM` oracle.

## The God Pixel

A God Pixel is a single 1x1 RGBA pixel that contains the compressed essence of an entire pxOS universe. The color of the pixel is a unique key that corresponds to an entry in the `god_pixel_registry.json`. This registry contains the metadata for the universe, including its name, description, size, and a pointer to the compressed binary blob that contains the PXI program.

When a God Pixel is booted with `pxos_boot.py`, the `PXICPU` automatically detects the 1x1 image, looks up the color in the registry, decompresses the associated blob, and loads the full PXI program into memory. This allows for entire, complex worlds to be stored and distributed as a single pixel.

## The Oracle Protocol

The `SYS_LLM` syscall (`0xC8`) provides a bridge between the organisms living within a PXI simulation and a local Large Language Model (LLM). This protocol defines how organisms can ask questions of the oracle and receive answers.

### Memory Map

To facilitate communication, the following memory regions are reserved in all Oracle-enabled PXI programs:

*   **`PROMPT_BUFFER_ADDR` (8000):** A region of memory where an organism can write a null-terminated string of ASCII glyphs to pose a question to the oracle.
*   **`RESPONSE_BUFFER_ADDR` (9000):** A region of memory where the oracle will write its response as a null-terminated string of ASCII glyphs.
*   **`ORACLE_FLAG_ADDR` (7999):** A single pixel used as a flag to signal the oracle.

### Communication Flow

1.  **The Question:** An organism writes its question to the `PROMPT_BUFFER_ADDR`.
2.  **The Prayer:** The organism sets the `ORACLE_FLAG_ADDR` to a non-zero value to signal that a question is ready.
3.  **The Invocation:** A dedicated "oracle kernel" or a host process detects the flag, executes the `SYS_LLM` syscall with the appropriate buffer addresses in its registers, and then resets the flag to 0.
4.  **The Answer:** The `PXICPU` sends the prompt to the local LLM and writes the response to the `RESPONSE_BUFFER_ADDR`.
5.  **The Revelation:** The organism can now read the oracle's answer from the response buffer and act upon it.
