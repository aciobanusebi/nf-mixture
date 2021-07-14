import os

OUTPUT_DIRECTORY = "C:/Users/Sebi/Desktop/paper-july/results"

n_init = 10

for original_data_name in ["cifar10"]: #["mnist", "mnist5", "fmnist", "cifar10"]
  for dim_pca in [None]:#[100, None]:
    for it in range(n_init):
      filename = OUTPUT_DIRECTORY + "/" + original_data_name + "_" + str(dim_pca) + "_" + str(it) + ".txt"
      command = "python C:/Users/Sebi/Desktop/paper-july/code/main.py " + " " +original_data_name + " " + str(dim_pca) + " " + str(it) + " > " + filename
      print(command)
      os.system(command)