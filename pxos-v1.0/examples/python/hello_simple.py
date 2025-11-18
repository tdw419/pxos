#!/usr/bin/env python3
"""
Simple Hello World for pxOS

This is the simplest possible pxOS program in Python.
It clears the screen and prints a message.
"""

from pxos import clear_screen, print_text, loop_forever


def main():
    clear_screen()
    print_text("Hello from pxOS Python!")
    loop_forever()


if __name__ == "__main__":
    main()
