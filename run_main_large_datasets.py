import os

OUTPUT_DIRECTORY = "../results"

n_init = 10

for original_data_name in ["mnist", "mnist5", "fmnist", "cifar10"]:
  directory = OUTPUT_DIRECTORY + "/" + original_data_name
  if not os.path.exists(directory):
    os.makedirs(directory)
  for dim_pca in [100, None]:
    for it in range(n_init):
      filename = directory + "_" + str(dim_pca) + "_" + str(it) + ".txt"
      command = "python main_large_datasets.py " + " " +original_data_name + " " + str(dim_pca) + " " + str(it) + " > " + filename
      print(command)
      os.system(command)