from pathlib import Path
import argparse


workspace = Path("~/.workspace").expanduser().as_posix()

""" global params """
MODEL_NAME = "PraNet"
# MODEL_NAME = "FCBFormer"
# MODEL_NAME = "PVT"

_DATA_TYPE = "2D_Polyp"
# _DATA_TYPE = "SUN-SEG"

EXP = f"{MODEL_NAME}_{_DATA_TYPE}"


# MAX_EPOCHS = 20  # SUN-SEG
# MAX_EPOCHS = 30  # PraNet
# MAX_EPOCHS = 200  # FCBFormer
MAX_EPOCHS = 100  # PVT


BATCH_SIZE = 16  # PraNet, PVT
# BATCH_SIZE = 8  # FCBFormer


def parser_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", default=BATCH_SIZE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="model name")
    parser.add_argument("--epoch", type=int, default=MAX_EPOCHS, help="epoch number")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs")
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min")
    parser.add_argument("--trainsize", type=int, default=352, help="training dataset size")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")

    args = parser.parse_args()
    return args
