#!/usr/bin/env python3
"""
Multi-line Hello World for pxOS

Demonstrates printing multiple lines of text.
"""

from pxos import clear_screen, print_text, loop_forever


def main():
    clear_screen()

    print_text("=================================")
    print_text("       pxOS Python Edition       ")
    print_text("=================================")
    print_text("")
    print_text("Welcome to pxOS!")
    print_text("")
    print_text("This OS was written in Python,")
    print_text("then compiled to primitives.")
    print_text("")
    print_text("System Ready.")

    loop_forever()


if __name__ == "__main__":
    main()
