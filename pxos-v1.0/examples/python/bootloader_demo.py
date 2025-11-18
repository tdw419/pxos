#!/usr/bin/env python3
"""
Bootloader Demo for pxOS

Simulates a simple bootloader sequence with status messages.
"""

from pxos import clear_screen, print_text, print_char, loop_forever
from pxos import make_color, WHITE, BLACK, GREEN, CYAN


def main():
    clear_screen()

    # Boot header
    print_text("pxOS Bootloader v1.0", attr=make_color(CYAN, BLACK))
    print_text("====================", attr=make_color(CYAN, BLACK))
    print_text("")

    # Boot sequence
    print_text("Initializing system...", attr=make_color(WHITE, BLACK))
    print_text("[OK]", attr=make_color(GREEN, BLACK))
    print_text("")

    print_text("Loading kernel...", attr=make_color(WHITE, BLACK))
    print_text("[OK]", attr=make_color(GREEN, BLACK))
    print_text("")

    print_text("Setting up memory...", attr=make_color(WHITE, BLACK))
    print_text("[OK]", attr=make_color(GREEN, BLACK))
    print_text("")

    print_text("Starting Python runtime...", attr=make_color(WHITE, BLACK))
    print_text("[OK]", attr=make_color(GREEN, BLACK))
    print_text("")

    print_text("")
    print_text("System boot complete!", attr=make_color(GREEN, BLACK))
    print_text("")
    print_text("Press any key to halt.", attr=make_color(WHITE, BLACK))

    loop_forever()


if __name__ == "__main__":
    main()
