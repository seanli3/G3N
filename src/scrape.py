import os
import re
import numpy as np

# collect info from zinc12k
configs = []
for t in [1,2,3]:
  for d in [1,2,3]:
    for l in [3,4,5]:
      configs.append(f'{t}_{d}_{l}')

directory = 'results/zinc12k/'
for config in configs:
  print('-------------------------------')
  print(config)
  tests = []
  trains = []
  for filename in os.listdir(os.path.join(directory, config)):
    f = os.path.join(directory, config, filename)
    for line in open(f).readlines():
      if "Test MAE" in line:
        tests.append(float(re.findall("\d+\.\d+", line)[0]))
      if "Train MAE" in line:
        trains.append(float(re.findall("\d+\.\d+", line)[0]))
  tests = np.array(tests)
  trains = np.array(trains)
  # print(f'test: {np.mean(tests):.3f} ± {np.std(tests):.3f}')
  # print(f'test: {np.mean(tests):.3f}\sd{{{np.std(tests):.3f}}}')
  # print(f'train: {np.mean(trains):.3f} ± {np.std(trains):.3f}')
  # print(f'test: {np.mean(trains):.3f}\sd{{{np.std(trains):.3f}}}')
  # print(f'gen_gap: {np.mean(tests - trains):.3f} ± {np.std(tests - trains):.3f}')
  print(f'gen_gap: {np.mean(tests - trains):.3f}\sd{{{np.std(tests - trains):.3f}}}')

# for config in configs:
#      print('-------------------------------')
#      print(benchmark)
#      for config in configs:
#           directory_in_str = f'results/zinc12k/{config}'
#           if not os.path.exists(directory_in_str):
#                continue
#           directory = os.fsencode(directory_in_str)
#           solved = 0
          
#           for file in os.listdir(directory):
#                filename = os.fsdecode(file)
#                filesize = os.path.getsize(f'{directory_in_str}/{filename}')


#                if 'true' in open(f'{directory_in_str}/{filename}').read():
#                     solved += 1

#           print(config, solved)