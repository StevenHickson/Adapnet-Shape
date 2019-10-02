from tensorboard import program
import time
import sys

tb = program.TensorBoard()
print('Launching tensorboard on port ' + str(sys.argv[1]))
tb.configure(argv=[None, '--logdir', './graphs/' + str(sys.argv[2]), '--port', sys.argv[1]])
tb.launch()
while(1):
    time.sleep(1)
