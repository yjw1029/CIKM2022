import argparse
from pathlib import Path
import re
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default="../amlt/cikm2022")
    parser.add_argument("--save-path", type=str, default="../amlt/merge_rslt")

    return parser.parse_args()


def parse_eval_rslt(file_name):
    with open(file_name, "r") as f:
        clients_rslt = {}
        for line in f:
            z = re.match("client (\d+) .*relative_impr: (-?\d.\d+e?-?\d+]?) \n", line)
            if z:
                client_num, best_impr = z.groups()
                clients_rslt[int(client_num)] = float(best_impr)
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



def main(args):
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

if __name__ == "__main__":
    args = parse_args()
    main(args)

