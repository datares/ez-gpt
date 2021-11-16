from config import config
from model import Model


def main():
    ckpt_path = config["checkpoint_path"]
    model = Model().load_from_checkpoint(ckpt_path)
    for i in range(10):
        model.generate()

if __name__ == "__main__":
    main()
