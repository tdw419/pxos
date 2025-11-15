#!/usr/bin/env python3
"""
god_pixel_cli.py - God Pixel Zoo Management CLI

A command-line tool to interact with the God Pixel registry.
"""

import json
from pathlib import Path
from PIL import Image
import argparse

REGISTRY_FILE = Path("god_pixel_registry.json")

def list_worlds():
    if not REGISTRY_FILE.exists():
        print("Registry not found.")
        return
    reg = json.loads(REGISTRY_FILE.read_text())
    for color, meta in reg.items():
        print(f"  - {meta['name']}: {meta['description']} (God Pixel: {color})")

def describe_world(name: str):
    if not REGISTRY_FILE.exists():
        print("Registry not found.")
        return
    reg = json.loads(REGISTRY_FILE.read_text())
    for color, meta in reg.items():
        if meta["name"].lower() == name.lower():
            print(f"World: {meta['name']}")
            print(f"  Description: {meta['description']}")
            print(f"  God Pixel (RGBA): {color}")
            print(f"  Resurrected Size: {meta['size'][0]}x{meta['size'][1]}")
            print(f"  Compressed Blob: {meta['blob']}")
            return
    print(f"World '{name}' not found in registry.")

def create_pixel(name: str, output_path: str):
    if not REGISTRY_FILE.exists():
        print("Registry not found.")
        return
    reg = json.loads(REGISTRY_FILE.read_text())
    for color, meta in reg.items():
        if meta["name"].lower() == name.lower():
            rgba = tuple(int(c) for c in color.split(','))
            img = Image.new("RGBA", (1, 1), rgba)
            img.save(output_path)
            print(f"God Pixel for '{name}' created at '{output_path}'")
            return
    print(f"World '{name}' not found in registry.")

def main():
    parser = argparse.ArgumentParser(description="God Pixel Zoo Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    parser_list = subparsers.add_parser("list", help="List all worlds in the zoo.")

    # describe command
    parser_describe = subparsers.add_parser("describe", help="Describe a specific world.")
    parser_describe.add_argument("name", help="Name of the world to describe.")

    # create-pixel command
    parser_create = subparsers.add_parser("create-pixel", help="Create a God Pixel for a world.")
    parser_create.add_argument("name", help="Name of the world.")
    parser_create.add_argument("output_path", help="Path to save the 1x1 God Pixel PNG.")

    args = parser.parse_args()

    if args.command == "list":
        list_worlds()
    elif args.command == "describe":
        describe_world(args.name)
    elif args.command == "create-pixel":
        create_pixel(args.name, args.output_path)

if __name__ == "__main__":
    main()
