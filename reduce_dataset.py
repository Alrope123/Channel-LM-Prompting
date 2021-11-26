import argparse
import os

def main(args):
    assert args.label_num < 6 and args.label_num >= 3 

    splits = ["train", "test"]
    output_dir = os.path.join(args.data_dir, args.task + "-" + str(args.label_num))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in splits:
        with open(os.path.join(args.data_dir, args.task, split + ".csv"), 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if int(line[0]) in range(args.label_num)]
        with open(os.path.join(output_dir, split + ".csv"), "w") as f:
            f.writelines(lines) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/original")
    parser.add_argument("--label_num", type=int, default=5)
    parser.add_argument("--task", type=str, default="trec")
    args = parser.parse_args()
    main(args)
