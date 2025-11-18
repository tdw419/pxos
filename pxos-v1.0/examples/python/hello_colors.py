#!/usr/bin/env python3
"""
Colored Hello World for pxOS

Demonstrates using color attributes.
"""

from pxos import clear_screen, print_text, loop_forever, make_color
from pxos import WHITE, BLUE, YELLOW, BLACK, GREEN, RED


def main():
    # Clear with blue background
    clear_screen(make_color(WHITE, BLUE))

    # Title bar
    print_text("  pxOS v1.0 - Python Edition  ", attr=make_color(YELLOW, BLUE))

    # Main content
    print_text("")
    print_text("Status: ", attr=make_color(WHITE, BLACK))
    print_text("ONLINE", attr=make_color(GREEN, BLACK))
    print_text("")
    print_text("Error Count: ", attr=make_color(WHITE, BLACK))
    print_text("0", attr=make_color(GREEN, BLACK))
    print_text("")
    print_text("This is a warning!", attr=make_color(YELLOW, BLACK))
    print_text("This is an error!", attr=make_color(RED, BLACK))

    loop_forever()


if __name__ == "__main__":
    main()
