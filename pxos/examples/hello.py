# pxos/examples/hello.py
from pxos import print_text, clear_screen

def main():
    clear_screen()
    print_text("HELLO PXOS FROM PYTHON", row=10, col=5)

if __name__ == "__main__":
    main()
