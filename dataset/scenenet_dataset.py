from helper import DatasetHelper
import numpy as np
import sys
sys.path.append('/nethome/shickson3/CreateNormals/')
from python.calc_normals import NormalCalculation

flat_labels = []
camera_params = [277.128129211,0,160,0,289.705627485,120,0,0,1]
normal_params = [5,0.02,10,0.04]

class ScenenetDataset(DatasetHelper):
    normal_calculator = NormalCalculation(camera_params, normal_params, flat_labels)

    def MapLabels(self, label):
        return label.astype(np.uint16)

