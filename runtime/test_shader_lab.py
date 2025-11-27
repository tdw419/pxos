import requests
import json
import time

def test_shader_lab():
    url = "http://localhost:5000/api/generate-shader"
    prompt = "draw a green circle"

    print(f"Sending prompt: '{prompt}' to {url}")

    try:
        response = requests.post(url, json={"prompt": prompt})

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Success!")
            print(f"WGSL Length: {len(data.get('wgsl', ''))}")
            framebuffer = data.get('framebuffer')
            print(f"Framebuffer size: {len(framebuffer)} pixels (Expected 1024)")

            # Basic validation of framebuffer content
            non_zero = sum(1 for p in framebuffer if p != 0)
            print(f"Non-zero pixels: {non_zero}")

            if non_zero > 0:
                print("PASS: Framebuffer contains data.")
            else:
                print("WARN: Framebuffer is all zeros (might be valid if shader draws black/transparent).")

        else:
            print("Failed.")
            print(response.text)

    except Exception as e:
        print(f"Connection Error: {e}")
        print("Make sure runtime/shader_lab.py is running!")

if __name__ == "__main__":
    test_shader_lab()
