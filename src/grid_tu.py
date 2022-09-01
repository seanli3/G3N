import os
import pathlib
import argparse
from datetime import datetime

# script for running grid search on tu datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bio', choices=["bio", "soc"])
parser.add_argument('--d', type=int, default=1,
                    help='distance of neighbourhood (default: 1)')
parser.add_argument('--t', type=int, default=2,
                    help='size of t-subsets (default: 2)')
args = parser.parse_args()

if args.dataset=="bio":
  datasets = ["PTC_MR","MUTAG","NCI1","PROTEINS",]
else:
  datasets = ["IMDB-BINARY","IMDB-MULTI"]

neighbourhood = [(args.t, args.d)]
hidden = [32, 64, 128]
batch_sizes = [32, 128]
dropouts = [0, 0.5]

for ds in datasets:
  for d, t in neighbourhood:
    for emb_dim in hidden:
      for batch_size in batch_sizes:
        for dropout in dropouts:
          cmd = f'python3 tu.py --dataset {ds} --d {d} --t {t} --drop_ratio {dropout} --emb_dim {emb_dim} --batch_size {batch_size} --device 1'
          print(cmd)

          results_dir = f"results/{ds}/"
          pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) 
          exp_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
          results_file = f'{results_dir}{cmd} {exp_date}.txt'

          result = os.popen(cmd).read()
          f = open(results_file, "w")
          f.write(result)
          f.close()
          print(result)

