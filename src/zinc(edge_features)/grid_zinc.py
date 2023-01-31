import os
import pathlib
from datetime import datetime

# script for running grid search on d, t, and num_layers on zinc for expressivity analysis

runs = 5
step = 20
configs = [  # (t, d, num_layers, emb_dim .s.t params~100k)
  # top 1
  (1,1,5,98),
  (1,1,4,105),
  (1,1,3,120),
  # top 2
  (1,2,5,78),
  (1,2,4,86),
  (1,2,3,98),
  # top 3
  (1,3,5,68),
  (1,3,4,76),
  (1,3,3,86),
  # top 2
  (2,1,5,78),
  (2,1,4,86),
  (2,1,3,98),
  # top 4
  (2,2,5,60),
  (2,2,4,68),
  (2,2,3,77),
  # top 6
  (2,3,5,50),
  (2,3,4,56),
  (2,3,3,66),
  # top 1
  (3,1,5,98),
  (3,1,4,105),
  (3,1,3,120),
  # top 5
  (3,2,5,54),
  (3,2,4,60),
  (3,2,3,70),
  # top 9
  (3,3,5,42),
  (3,3,4,46),
  (3,3,3,55),
]

for t,d,num_layer,emb_dim in configs:
  for _ in range(runs):
    cmd = f'python3 zinc.py --t {t} --d {d} --num_layer {num_layer} --emb_dim {emb_dim} --step {step}'
    print(cmd)

    results_dir = f"results/zinc12k/{t}_{d}_{num_layer}/"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) 
    exp_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    results_file = f'{results_dir}{t}_{d}_{num_layer} {exp_date}.txt'

    result = os.popen(cmd).read()
    f = open(results_file, "w")
    f.write(result)
    f.close()
    print(result)

