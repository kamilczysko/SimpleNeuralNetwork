import numpy as np
import glob
import os
import random as r

path = 'data/prep/'
prep = []
learn_data_array = []
learn_target_array = []

path, sth, files = next(os.walk(path))
num_of_files = len(files)
answer_array = []

def load(file):
    data = np.load(file)
    return data

def prep_answer_array(element):
    target = [0]*num_of_files
    target[len(answer_array)] = 1
    answer_array.append([element, np.argmax(target)])
    return target

for filename in glob.glob(os.path.join(path, '*.npy')):
    base_name = os.path.basename(filename).split('.')[0]
    target = prep_answer_array(base_name)
    array = load(filename)
    for item in array:
        prep.append([item, target])

r.shuffle(prep)
for item in prep:
    learn_data_array.append(item[0])
    learn_target_array.append(item[1])
