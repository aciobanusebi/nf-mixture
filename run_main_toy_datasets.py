import os

OUTPUT_DIRECTORY = "results"

n_init = 10

for original_data_name in ["two_banana","smile","moons","circles","pinwheel"]: 
  for it in range(n_init):
    directory = OUTPUT_DIRECTORY + "/" +  original_data_name
    if not os.path.exists(directory):
      os.makedirs(directory)
    filename = directory + "/" + str(it) + ".txt"
    command = "python main_toy_datasets.py " + " " + original_data_name + " "  + str(it) + " " + directory + " > " + filename
    print(command)
    os.system(command)
