import os
import logging
from pathlib import Path
import random
import numpy as np
import torch

from param import parse_args
from utils import setuplogger
from trainer import get_trainer_cls

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    logging.info("[+] start init trainer.")
    trainer_cls = get_trainer_cls(args.trainer_cls)
    trainer = trainer_cls(args)
    logging.info("[-] end init trainer.")

    logging.info("[+] start running")
    trainer.run()
    logging.info("[-] end running")


if __name__ == "__main__":
    args = parse_args()
    out_path = Path(args.out_path) 
    data_path = Path(args.data_path)

    out_path.mkdir(exist_ok=True, parents=True)

    setuplogger(args)

    logging.info(args)

    torch.cuda.set_device(int(args.device))
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    main(args)