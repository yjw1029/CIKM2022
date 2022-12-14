import argparse
from pathlib import Path
import re
from datetime import datetime
import pytz
from collections import defaultdict
import numpy as np
import logging
import sys


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
    parser.add_argument("--out-path", type=str, default="../amlt/cikm2022")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="merge",
        choices=["merge", "append", "kfold", "kfold2"],
    )
    parser.add_argument("--clients", type=int, nargs="+", default=list(range(1, 14)))

    # for append
    parser.add_argument("--pre-merge-path", type=str, default=None)
    parser.add_argument("--append-src", type=str, default=None)
    parser.add_argument("--append-clients", type=int, nargs="+", default=None)

    # ensemble topk
    parser.add_argument("--topk_min", type=int, default=0)
    parser.add_argument("--topk_max", type=int, default=None)
    parser.add_argument("--topk_indices", type=int, nargs="+", default=None)

    # weight
    parser.add_argument("--use-weight", type=str2bool, default=False)

    # soft
    parser.add_argument("--soft", type=str2bool, default=False)

    # k_fold
    parser.add_argument("--k-fold", type=int, default=4)
    parser.add_argument("--k-fold-avg", type=str2bool, default=False)

    return parser.parse_args()


def parse_eval_rslt(file_name):
    with open(file_name, "r") as f:
        clients_rslt = {}
        for line in f:
            z = re.match("client (\d+) .*relative_impr: (-?\d.\d+e?-?\d+]?) \n", line)
            if z:
                client_id, best_impr = z.groups()
                clients_rslt[int(client_id)] = float(best_impr)
    return clients_rslt


def parse_merge_rslt(file_name):
    with open(file_name, "r") as f:
        clients_rslt = defaultdict(list)
        for line in f:
            client_id, task_name, impr = line.strip("\n").split(", ")
            clients_rslt[int(client_id)].append((task_name, float(impr)))
    return clients_rslt


def sort_task_by_impr(task_rslts):
    sorted_rslt_tasks = {}
    for uid in clients:
        rslt_task_pairs = []
        for task in task_rslts:
            if uid in task_rslts[task]:
                rslt_task_pairs.append((task_rslts[task][uid], task))

        rslt_task_pairs.sort(key=lambda x: x[0], reverse=True)

        sorted_rslt_tasks[uid] = rslt_task_pairs
    return sorted_rslt_tasks


def merge_best_rslt(out_path, sorted_rslt_tasks, save_path):
    merged_lines = []
    for uid in clients:
        selected_task = sorted_rslt_tasks[uid][0]

        impr_rslt, task_name = selected_task
        with open(out_path / task_name / "prediction_test.csv", "r") as f:
            for line in f:
                u, sid = line.split(",")[:2]
                if int(u) == uid:
                    merged_lines.append(line)

        with open(save_path / "merged_task.txt", "a") as f:
            f.write(f"{uid}, {task_name}, {impr_rslt}\n")

    with open(save_path / "prediction_test.csv", "w") as f:
        f.writelines(merged_lines)


def merge_cls_rslt(lines, weights, use_weight=False):
    # uid, sid, cls
    uids, sids, preds = [], [], []
    merged_pred = defaultdict(float)
    for cnt, line in enumerate(lines):
        uid, sid, pred = line.strip("\n").split(",")
        uids.append(uid)
        sids.append(sid)
        if use_weight:
            merged_pred[pred] += weights[cnt]
        else:
            merged_pred[pred] += 1.0

    assert len(set(uid)) == 1 and len(set(sids)) == 1
    merged_pred = max(merged_pred.items(), key=lambda k: k[1])[0]
    merged_line = f"{uid},{sid},{merged_pred}\n"
    return merged_line


def softmax(x):
    x = np.clip(x, -200, 200)
    x = np.exp(x)
    return x / np.sum(x)


def merge_cls_rslt_soft(
    lines, weights, use_weight=False, apply_softmax=True, return_soft=False
):
    # uid, sid, cls
    uids, sids = [], []
    merged_pred = defaultdict(float)
    for cnt, line in enumerate(lines):
        uid, sid = line.strip("\n").split(",")[:2]
        pred = np.array(
            [float(i) for i in line.strip("\n").split(",")[2:]], dtype=np.float64
        )
        if apply_softmax:
            pred = softmax(pred)

        uids.append(uid)
        sids.append(sid)
        for i, value in enumerate(pred):
            if use_weight:
                merged_pred[i] += value * weights[cnt]
            else:
                merged_pred[i] += value

    assert len(set(uid)) == 1 and len(set(sids)) == 1
    if return_soft:
        merged_pred = [merged_pred[i] / len(lines) for i in range(len(merged_pred))]
        merged_pred = ",".join([str(i) for i in merged_pred])
    else:
        merged_pred = max(merged_pred.items(), key=lambda k: k[1])[0]
    merged_line = f"{uid},{sid},{merged_pred}\n"
    return merged_line


def merge_binary_cls_soft(
    out_path,
    uid,
    selected_tasks,
    use_weight=False,
    apply_softmax=True,
    return_soft=False,
    suffix=""
):
    task_lines = {}
    task_rslts = []
    test_rslts_dict = {}
    for selected_task in selected_tasks:
        impr_rslt, task_name = selected_task
        task_lines[selected_task] = {}
        with open(out_path / task_name / f"prediction_soft_test{suffix}.csv", "r") as f:
            for line in f:
                if int(line.split(",")[0]) == uid:
                    task_lines[selected_task][int(line.split(",")[1])] = line
        test_rslts_dict[selected_task] = impr_rslt
        task_rslts.append(f"{uid}, {task_name}, {impr_rslt}\n")

    merged_lines = []
    for i in range(len(task_lines[selected_tasks[0]])):
        lines = [task_lines[t][i] for t in selected_tasks]
        weights = [test_rslts_dict[t] for t in selected_tasks]
        merged_lines.extend(
            merge_cls_rslt_soft(
                lines,
                weights,
                use_weight,
                apply_softmax=apply_softmax,
                return_soft=return_soft,
            )
        )
    return merged_lines, task_rslts


def merge_binary_cls(out_path, uid, selected_tasks, use_weight=False):
    task_lines = {}
    task_rslts = []
    test_rslts_dict = {}
    for selected_task in selected_tasks:
        impr_rslt, task_name = selected_task
        task_lines[selected_task] = {}
        with open(out_path / task_name / "prediction_test.csv", "r") as f:
            for line in f:
                if int(line.split(",")[0]) == uid:
                    task_lines[selected_task][int(line.split(",")[1])] = line
        test_rslts_dict[selected_task] = impr_rslt
        task_rslts.append(f"{uid}, {task_name}, {impr_rslt}\n")

    merged_lines = []
    for i in range(len(task_lines[selected_tasks[0]])):
        lines = [task_lines[t][i] for t in selected_tasks]
        weights = [test_rslts_dict[t] for t in selected_tasks]
        merged_lines.extend(merge_cls_rslt(lines, weights, use_weight))
    return merged_lines, task_rslts


def merge_regression_rslt(lines, weights, use_weight=False):
    # uid, sid, cls
    uids, sids, preds = [], [], []
    for line in lines:
        uid, sid = line.strip("\n").split(",")[:2]
        pred = np.array([float(i) for i in line.strip("\n").split(",")[2:]])
        uids.append(uid)
        sids.append(sid)
        preds.append(pred)

    assert len(set(uids)) == 1 and len(set(sids)) == 1
    if use_weight:
        norm_weights = np.array(weights)
        norm_weights = np.expand_dims(norm_weights / norm_weights.sum(), axis=-1)
        merged_pred = np.sum(norm_weights * np.stack(preds, axis=0), axis=0)
    else:
        merged_pred = np.mean(np.stack(preds, axis=0), axis=0)
    merged_pred = [uid, sid] + list(merged_pred)
    merged_line = ",".join([str(_) for _ in merged_pred]) + "\n"
    return merged_line


def merge_regression(out_path, uid, selected_tasks, use_weight=False, suffix=""):
    task_lines = {}
    task_rslts = []
    test_rslts_dict = {}
    for selected_task in selected_tasks:
        impr_rslt, task_name = selected_task
        task_lines[selected_task] = {}
        with open(out_path / task_name / f"prediction_soft_test{suffix}.csv", "r") as f:
            for line in f:
                if int(line.split(",")[0]) == uid:
                    task_lines[selected_task][int(line.split(",")[1])] = line
        test_rslts_dict[selected_task] = impr_rslt
        task_rslts.append(f"{uid}, {task_name}, {impr_rslt}\n")

    merged_lines = []
    for i in range(len(task_lines[selected_tasks[0]])):
        lines = [task_lines[t][i] for t in selected_tasks]
        weights = [test_rslts_dict[t] for t in selected_tasks]
        merged_lines.extend(merge_regression_rslt(lines, weights, use_weight))

    return merged_lines, task_rslts


def merge_topk_rslt(
    out_path,
    sorted_rslt_tasks,
    save_path,
    k_min=0,
    k_max=10,
    k_indices=None,
    use_weight=False,
    soft=False,
    apply_softmax=True,
):
    merged_lines = []
    task_rslts = []
    for uid in clients:
        if k_max is None:
            selected_tasks = [sorted_rslt_tasks[uid][i] for i in k_indices]
        else:
            selected_tasks = sorted_rslt_tasks[uid][k_min:k_max]

        if uid <= 8:
            if soft:
                logging.info(f"soft merge client {uid} results.")
                merged_uid_lines, task_uid_rslts = merge_binary_cls_soft(
                    out_path,
                    uid,
                    selected_tasks,
                    use_weight=use_weight,
                    apply_softmax=apply_softmax,
                )
            else:
                logging.info(f"hard merge client {uid} results.")
                merged_uid_lines, task_uid_rslts = merge_binary_cls(
                    out_path, uid, selected_tasks, use_weight=use_weight
                )

        else:
            logging.info(f"average client {uid} results.")
            merged_uid_lines, task_uid_rslts = merge_regression(
                out_path, uid, selected_tasks, use_weight=use_weight
            )

        merged_lines.extend(merged_uid_lines)
        task_rslts.extend(task_uid_rslts)

    with open(save_path / "merged_task.txt", "a") as f:
        f.writelines(task_rslts)

    with open(save_path / "prediction_test.csv", "w") as f:
        f.writelines(merged_lines)


def merge(args):
    if args.save_path is None:
        tz_NY = pytz.timezone("Asia/Shanghai")
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge/merge_rslt_" + current_time

    out_path = Path(args.out_path)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    task_rslts = {}
    for task_path in out_path.iterdir():
        task_name = task_path.name
        if (task_path / "eval_rslt.txt").exists():
            task_rslts[task_name] = parse_eval_rslt(task_path / "eval_rslt.txt")

    sorted_rslt_tasks = sort_task_by_impr(task_rslts)
    if args.topk_max is None:
        merge_best_rslt(out_path, sorted_rslt_tasks, save_path)
    else:
        logging.info(f"best {args.topk_min} to {args.topk_max} results.")
        merge_topk_rslt(
            out_path,
            sorted_rslt_tasks,
            save_path,
            k_min=args.topk_min,
            k_max=args.topk_max,
            k_indices=args.topk_indices,
            soft=args.soft,
            use_weight=args.use_weight,
            apply_softmax=False,
        )


def append(args):
    if args.save_path is None:
        tz_NY = pytz.timezone("Asia/Shanghai")
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge/merge_rslt_" + current_time

    pre_merge_path = Path(args.pre_merge_path)
    append_src = Path(args.append_src)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    merged_lines = []
    for uid in clients:
        if uid not in args.append_clients:
            with open(pre_merge_path / "prediction_test.csv", "r") as f:
                for line in f:
                    u, sid = line.split(",")[:2]
                    if int(u) == uid:
                        merged_lines.append(line)
        else:
            with open(append_src / "prediction_test.csv", "r") as f:
                for line in f:
                    u, sid = line.split(",")[:2]
                    if int(u) == uid:
                        merged_lines.append(line)

    with open(save_path / "prediction_test.csv", "w") as f:
        f.writelines(merged_lines)

    merged_rslts = []
    pre_merged_rslts = parse_merge_rslt(pre_merge_path / "merged_task.txt")

    if (append_src / "eval_rslt.txt").exists():
        append_rslts = defaultdict(list)
        client_rslts = parse_eval_rslt(append_src / "eval_rslt.txt")
        impr_rslt = client_rslts[int(uid)]
        task_name = append_src.name
        for uid in client_rslts:
            append_rslts[int(uid)].append((task_name, float(impr_rslt)))
    else:
        append_rslts = parse_merge_rslt(append_src / "merged_task.txt")

    with open(pre_merge_path / "merged_task.txt", "r") as f:
        for uid in range(1, 14):
            if int(uid) in args.append_clients:
                rslts = append_rslts[uid]
            else:
                rslts = pre_merged_rslts[uid]

            for task_name, impr_rslt in rslts:
                merged_rslts.append(f"{uid}, {task_name}, {impr_rslt}\n")

    with open(save_path / "merged_task.txt", "a") as f:
        f.writelines(merged_rslts)


def merge_k_fold_rslt(task_rslts, k_fold):

    merge_rslt_tasks = {}
    for task in task_rslts:
        merge_rslt_tasks[task] = {}
        for uid in clients:
            k_fold_rslt = []
            for i in range(args.k_fold):
                if i in task_rslts[task] and uid in task_rslts[task][i]:
                    k_fold_rslt.append(task_rslts[task][i][uid])
            merge_rslt_tasks[task][uid] = np.mean(k_fold_rslt)

    return merge_rslt_tasks


def merge_k_fold_pred(task_path, k_fold, soft=True):
    task_lines = {}
    for val_fold in range(k_fold):
        task_lines[val_fold] = {}
        if (task_path / f"prediction_soft_test_{val_fold}.csv").exists():
            with open(task_path / f"prediction_soft_test_{val_fold}.csv", "r") as f:
                for line in f:
                    uid = int(line.split(",")[0])
                    task_lines[val_fold][(uid, int(line.split(",")[1]))] = line

    merged_lines = []
    keys = list(task_lines[0].keys())
    for i in range(len(task_lines[0])):
        uid, sid = keys[i]
        lines = [
            task_lines[val_fold][keys[i]]
            for val_fold in range(k_fold)
            if val_fold in task_lines and keys[i] in task_lines[val_fold]
        ]
        if soft:
            if uid <= 8:
                merged_lines.extend(
                    merge_cls_rslt_soft(
                        lines,
                        weights=None,
                        use_weight=False,
                        apply_softmax=True,
                        return_soft=True,
                    )
                )
            else:
                merged_lines.extend(
                    merge_regression_rslt(lines, weights=None, use_weight=False)
                )
        else:
            if uid <= 8:
                merged_lines.extend(
                    merge_cls_rslt_soft(
                        lines,
                        weights=None,
                        use_weight=False,
                        apply_softmax=True,
                        return_soft=False,
                    )
                )
            else:
                merged_lines.extend(
                    merge_regression_rslt(lines, weights=None, use_weight=False)
                )

    with open(task_path / "prediction_soft_test.csv", "w") as f:
        f.writelines(merged_lines)


def kfold(args):
    if args.save_path is None:
        tz_NY = pytz.timezone("Asia/Shanghai")
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge/merge_rslt_" + current_time

    out_path = Path(args.out_path)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    task_rslts = {}
    for task_path in out_path.iterdir():
        task_name = task_path.name

        if args.k_fold_avg:
            merge_k_fold_pred(task_path, args.k_fold)

        task_rslts[task_name] = {}
        for i in range(args.k_fold):
            if (task_path / f"eval_rslt_{i}.txt").exists():
                task_rslts[task_name][i] = parse_eval_rslt(
                    task_path / f"eval_rslt_{i}.txt"
                )

    merge_task_rlsts = merge_k_fold_rslt(task_rslts, args.k_fold)
    sorted_rslt_tasks = sort_task_by_impr(merge_task_rlsts)
    if args.topk_max is None and args.topk_indices is None:
        merge_best_rslt(out_path, sorted_rslt_tasks, save_path)
    else:
        if args.topk_max is None:
            logging.info(f"best {args.topk_indices} results.")
        else:
            logging.info(f"best {args.topk_min} to {args.topk_max} results.")
        merge_topk_rslt(
            out_path,
            sorted_rslt_tasks,
            save_path,
            k_min=args.topk_min,
            k_max=args.topk_max,
            k_indices=args.topk_indices,
            soft=args.soft,
            use_weight=args.use_weight,
            apply_softmax=False,
        )


def merge_topk_rslt_kfold2(
    out_path,
    sorted_rslt_folds,
    save_path,
    k_min=0,
    k_max=10,
    k_indices=None,
    use_weight=False,
    apply_softmax=True,
):

    for fold in sorted_rslt_folds:

        merged_lines = []
        task_rslts = []

        sorted_rslt_tasks = sorted_rslt_folds[fold]
        for uid in clients:
            if k_max is None:
                selected_tasks = [sorted_rslt_tasks[uid][i] for i in k_indices]
            else:
                selected_tasks = sorted_rslt_tasks[uid][k_min:k_max]

            if uid <= 8:
                logging.info(f"soft merge client {uid} results.")
                merged_uid_lines, task_uid_rslts = merge_binary_cls_soft(
                    out_path,
                    uid,
                    selected_tasks,
                    use_weight=use_weight,
                    apply_softmax=apply_softmax,
                    return_soft=True,
                    suffix=f"_{fold}"
                )
            else:
                logging.info(f"average client {uid} results.")
                merged_uid_lines, task_uid_rslts = merge_regression(
                    out_path, uid, selected_tasks, use_weight=use_weight, suffix=f"_{fold}"
                )

            merged_lines.extend(merged_uid_lines)
            task_rslts.extend(task_uid_rslts)

        with open(save_path / f"merged_task_{fold}.txt", "a") as f:
            f.writelines(task_rslts)

        with open(save_path / f"prediction_soft_test_{fold}.csv", "w") as f:
            f.writelines(merged_lines)

    task_lines = {}
    for val_fold in sorted_rslt_folds:
        task_lines[val_fold] = {}
        if (save_path / f"prediction_soft_test_{val_fold}.csv").exists():
            with open(save_path / f"prediction_soft_test_{val_fold}.csv", "r") as f:
                for line in f:
                    uid = int(line.split(",")[0])
                    task_lines[val_fold][(uid, int(line.split(",")[1]))] = line

    merged_lines = []
    keys = list(task_lines[0].keys())
    for i in range(len(task_lines[0])):
        uid, sid = keys[i]
        lines = [
            task_lines[val_fold][keys[i]]
            for val_fold in sorted_rslt_folds
            if val_fold in task_lines and keys[i] in task_lines[val_fold]
        ]

        if uid <= 8:
            merged_lines.extend(
                merge_cls_rslt_soft(
                    lines,
                    weights=None,
                    use_weight=use_weight,
                    apply_softmax=False,
                    return_soft=False,
                )
            )
        else:
            merged_lines.extend(
                merge_regression_rslt(lines, weights=None, use_weight=False)
            )

    with open(save_path / "prediction_test.csv", "w") as f:
        f.writelines(merged_lines)


def kfold2(args):
    # sort each fold first and merg
    if args.save_path is None:
        tz_NY = pytz.timezone("Asia/Shanghai")
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge/merge_rslt_" + current_time

    out_path = Path(args.out_path)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    task_rslts = {}
    for i in range(args.k_fold):
        task_rslts[i] = {}
        for task_path in out_path.iterdir():
            task_name = task_path.name
            if (task_path / f"eval_rslt_{i}.txt").exists():
                task_rslts[i][task_name] = parse_eval_rslt(
                    task_path / f"eval_rslt_{i}.txt"
                )

    sorted_rslt_tasks = {}
    for i in task_rslts:
        sorted_rslt_tasks[i] = sort_task_by_impr(task_rslts[i])

    merge_topk_rslt_kfold2(
        out_path,
        sorted_rslt_tasks,
        save_path,
        k_min=args.topk_min,
        k_max=args.topk_max,
        use_weight=args.use_weight,
        apply_softmax=True,
    )


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(f"[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


if __name__ == "__main__":
    args = parse_args()

    setuplogger()
    logging.info(args)
    clients = args.clients

    if args.mode == "merge":
        merge(args)
    if args.mode == "append":
        append(args)
    if args.mode == "kfold":
        kfold(args)
    if args.mode == "kfold2":
        kfold2(args)
