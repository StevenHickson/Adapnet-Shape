import os
import sys

valid_checkpoints = [49999, 99999, 150000]

#ex model.ckpt-81999.data-00000-of-00001

for (dirpath, dirname, filenames) in os.walk(sys.argv[1]):
    try:
        # Do this to throw errors for thinks that don't have model.ckpt
        for filename in filenames:
            no_op = filename.split('model')[1]
            file_num = int(filename.split('-')[1].split('.')[0])
            if file_num not in valid_checkpoints:
                os.remove(os.path.join(dirpath,filename))
                print('removing filename ' + os.path.join(dirpath,filename))
    except:
        pass
