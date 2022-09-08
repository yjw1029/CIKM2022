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

    parser.add_argument("--run-name", type=str, default="")

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
    parser.add_argument("--patient", type=int, default=None)

    parser.add_argument("--local-epoch", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1024)

    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--local-lr", type=float, default=None)

    parser.add_argument("--weight-decay", type=float, default=None)

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
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--num-bases", type=int, default=None)
    parser.add_argument(
        "--base-agg", type=str, default=None, choices=["decomposition", "moe"]
    )

    # finetune
    parser.add_argument("--enable-finetune", type=str2bool, default=False)
    parser.add_argument("--ft-local-optim-cls", type=str, default=None)
    parser.add_argument("--ft-lr", type=float, default=None)
    parser.add_argument("--max-ft-steps", type=int, default=None)
    parser.add_argument("--ft-local-epoch", type=int, default=None)

    # fl-reconstruction
    parser.add_argument("--reco-steps", type=int, default=None)

    # kfold
    parser.add_argument("--k-fold", type=int, default=4)
    parser.add_argument("--val-fold", type=int, default=0)

    return parser.parse_args()
