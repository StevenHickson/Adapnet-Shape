import os
import sys

valid_checkpoints = [49999, 99999, 150000]

#ex model.ckpt-81999.data-00000-of-00001

for filename in os.listdir(sys.argv[1]):
    try:
        # Do this to throw errors for thinks that don't have model.ckpt
        no_op = filename.split('model')[1]
        file_num = int(filename.split('-')[1].split('.')[0])
        if file_num not in valid_checkpoints:
            os.remove(os.path.join(sys.argv[1],filename))
    except:
        print('Ignoring ' + filename)

