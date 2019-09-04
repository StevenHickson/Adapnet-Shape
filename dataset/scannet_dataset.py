from helper import DatasetHelper
import numpy as np
import sys
import csv
sys.path.append('/nethome/shickson3/CreateNormals/')
from python.calc_normals import NormalCalculation

camera_params = [577.591,0,318.905,0,578.73,242.684,0,0,1]
normal_params = [5,0.02,30,0.04]
flat_labels = [1,2,7,8,9,11,19,22,24,29,30,32]

convert_to_20 = [0,1,2,3,4,5,6,7,8,9,10,11,12,9,13,20,14,20,4,20,2,0,0,0,15,20,0,0,16,0,20,0,20,17,18,20,19,0,20,20,0]
label_nyu_mapping = dict()
label_nyu_mapping[0] = 0
with open('/srv/datasets/scannet/scannetv2-labels.combined.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    start = True
    for row in reader:
        if not start:
            label_nyu_mapping[int(row[0])] = convert_to_20[int(row[4])]
        start = False

class ScannetDataset(DatasetHelper):
    normal_calculator = NormalCalculation(camera_params, normal_params, flat_labels)

    def MapLabels(self, label):
        label_nyu = np.array([label_nyu_mapping[x] for x in label.flatten()])
        return label_nyu.reshape(label.shape).astype(np.uint16)


