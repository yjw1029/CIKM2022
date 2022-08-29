import argparse
from pathlib import Path
import re
from datetime import datetime
import pytz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default="../amlt/cikm2022")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="merge", choices=["merge", "append"])

    # for append
    parser.add_argument("--pre-merge-path", type=str, default=None)
    parser.add_argument("--append-src", type=str, default=None)
    parser.add_argument("--append-clients", type=int, nargs="+", default=None)

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

def sort_task_by_impr(task_rslts):
    sorted_rslt_tasks = {}
    for uid in range(1, 14):
        rslt_task_pairs = []
        for task in task_rslts:
            if uid in task_rslts[task]:
                rslt_task_pairs.append((task_rslts[task][uid], task))
        
        rslt_task_pairs.sort(key=lambda x: x[0], reverse=True)
    
        sorted_rslt_tasks[uid] = rslt_task_pairs
    return sorted_rslt_tasks

def merge_best_rslt(out_path, sorted_rslt_tasks, save_path):
    merged_lines = []
    for uid in range(1, 14):   
        selected_task = sorted_rslt_tasks[uid][0]

        impr_rslt, task_name = selected_task
        with open(out_path / task_name / "prediction.csv", "r") as f:
            for line in f:
                u, sid = line.split(",")[:2]
                if int(u) == uid:
                    merged_lines.append(line)

        with open(save_path / "merged_task.txt", 'a') as f:
            f.write(f"{uid}, {task_name}, {impr_rslt}\n")
    
    with open(save_path / "prediction.csv", 'w') as f:
        f.writelines(merged_lines)

def merge(args):
    if args.save_path is None:
        tz_NY = pytz.timezone('Asia/Shanghai')
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge_rslt_" + current_time

    out_path = Path(args.out_path)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    task_rslts = {}
    for task_path in out_path.iterdir():
        task_name = task_path.name
        if (task_path / "eval_rslt.txt").exists():
            task_rslts[task_name] = parse_eval_rslt(task_path / "eval_rslt.txt")

    sorted_rslt_tasks = sort_task_by_impr(task_rslts)
    merge_best_rslt(out_path, sorted_rslt_tasks, save_path)


def append(args):
    if args.save_path is None:
        tz_NY = pytz.timezone('Asia/Shanghai')
        now = datetime.now(tz_NY)
        current_time = now.strftime("%Y%m%d%H%M")
        args.save_path = "../amlt/merge_rslt_" + current_time

    pre_merge_path = Path(args.pre_merge_path)
    append_src = Path(args.append_src)
    save_path = Path(args.save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    merged_lines = []
    for uid in range(1, 14):
        if uid not in args.append_clients:
            with open(pre_merge_path / "prediction.csv", "r") as f:
                for line in f:
                    u, sid = line.split(",")[:2]
                    if int(u) == uid:
                        merged_lines.append(line)
        else:
            with open(append_src / "prediction.csv", "r") as f:
                for line in f:
                    u, sid = line.split(",")[:2]
                    if int(u) == uid:
                        merged_lines.append(line)
    
    with open(save_path / "prediction.csv", 'w') as f:
        f.writelines(merged_lines)


    merged_rslts = []
    with open(pre_merge_path / "merged_task.txt", 'r') as f:
        for line in f:
            uid, task_name, impr_rslt = line.strip("\n").split(", ")
            if int(uid) in args.append_clients:
                client_rslts = parse_eval_rslt(append_src / "eval_rslt.txt")
                impr_rslt = client_rslts[int(uid)]
                task_name = append_src.name
                merged_rslts.append(f"{uid}, {task_name}, {impr_rslt}\n")
            else:
                merged_rslts.append(line)

    with open(save_path / "merged_task.txt", 'a') as f:
        f.writelines(merged_rslts)
    

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "merge":
        merge(args)
    if args.mode == "append":
        append(args)
