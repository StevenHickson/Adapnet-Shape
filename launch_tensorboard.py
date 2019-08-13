from tensorboard import program
import time
import sys

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './', '--port', sys.argv[1]])
tb.launch()
while(1):
    time.sleep(1)
