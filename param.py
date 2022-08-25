import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    # wandb initilization
    parser.add_argument("--enable-wandb", type=str2bool, default=False)
    parser.add_argument("--wandb-project", type=str, default="fedmoe", required=False)
    parser.add_argument("--wandb-api-key", type=str, default="")
    parser.add_argument("--wandb-run", type=str, default="", required=False)

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--client-sample-seed", type=int, default=1)

    parser.add_argument(
        "--data-path",
        type=str,
        default=os.getenv("AMLT_DATA_DIR", "../data"),
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=os.getenv("AMLT_OUTPUT_DIR", "../model"),
    )

    parser.add_argument("--trainer-cls", type=str, default=None)
    parser.add_argument("--server-cls", type=str, default=None)
    parser.add_argument("--client-cls", type=str, default=None)
    parser.add_argument("--agg-cls", type=str, default=None)
    parser.add_argument("--local-optim-cls", type=str, default=None)
    parser.add_argument("--global-optim-cls", type=str, default=None)

    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--train-log-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patient", type=int, default=10)

    parser.add_argument("--local-epoch", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1024)

    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--local-lr", type=float, default=None)

    parser.add_argument("--clients", type=int, nargs="+", default=list(range(1, 14)))
    parser.add_argument("--clients-per-step", type=int, default=13)
    parser.add_argument(
        "--client-config-file", type=str, default="./config/local_train_per_client.yaml"
    )
    parser.add_argument("--major-metric", type=str, default=None)

    parser.add_argument(
        "--param-filter-list",
        type=str,
        nargs="+",
        default=["encoder_atom", "encoder", "clf"],
    )

    # model parameters
    parser.add_argument("--model-cls", type=str, default=None)
    parser.add_argument(
        "--max-depth", type=int, default=None
    )
    parser.add_argument(
        "--hidden", type=int, default=None
    )
    parser.add_argument(
        "--dropout", type=float, default=None
    )
    parser.add_argument(
        "--pooling", type=str, default=None
    )

    # finetune 
    parser.add_argument(
        "--enable-finetune", type=str2bool, default=False
    )
    parser.add_argument(
        "--ft-local-optim-cls", type=str, default=None
    )
    parser.add_argument(
        "--ft-lr", type=float, default=None
    )

    return parser.parse_args()
