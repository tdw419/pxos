from PIL import Image

def verify_output(image_path):
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # Check the target pixel
    target_pixel = img.getpixel((8, 8))
    if target_pixel != (255, 255, 0, 255):
        print(f"Verification failed: Pixel at (8, 8) is {target_pixel}, not yellow.")
        return

    # Check that all other pixels are black
    for y in range(height):
        for x in range(width):
            if (x, y) != (8, 8):
                if img.getpixel((x, y)) != (0, 0, 0, 255):
                    print(f"Verification failed: Pixel at ({x}, {y}) is not black.")
                    return

    print("Verification successful: A single yellow pixel was found at (8, 8).")

if __name__ == "__main__":
    verify_output("pxi_hello_frame.png")
