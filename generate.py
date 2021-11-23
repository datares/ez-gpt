from model import Model
import sys

def main():
    ckpt_path = sys.argv[1]
    model = Model().load_from_checkpoint(ckpt_path)
    for _ in range(10):
        model.generate()


if __name__ == "__main__":
    main()
