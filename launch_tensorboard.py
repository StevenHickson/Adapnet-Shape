from tensorboard import program
import time

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', './', '--port', '9997'])
tb.launch()
while(1):
    time.sleep(1)
