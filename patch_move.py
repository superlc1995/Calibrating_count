import numpy as np
import shutil
import os 

labeled_list = np.loadtxt('label_list/sha_5.txt', dtype = str)
all_croped = os.listdir('part_A/uncertain_data/')

for file in all_croped:
    if (file.split('_')[0] + '_' + file.split('_')[1] + '.jpg') in labeled_list:
        shutil.copyfile('uncertain_data/' + file, 'uncertain_data_5/' + file )


labeled_list = np.loadtxt('label_list/sha_10.txt', dtype = str)

for file in all_croped:
    if (file.split('_')[0] + '_' + file.split('_')[1] + '.jpg') in labeled_list:
        shutil.copyfile('uncertain_data/' + file, 'uncertain_data_10/' + file )

labeled_list = np.loadtxt('label_list/sha_40.txt', dtype = str)

for file in all_croped:
    if (file.split('_')[0] + '_' + file.split('_')[1] + '.jpg') in labeled_list:
        shutil.copyfile('uncertain_data/' + file, 'uncertain_data_40/' + file )