import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import time

class ProcessPlotter(object):
    def __init__(self, freq, title, label):
        self.x = []
        self.y1 = []
        self.y2 = []
        self.y3 = []
        self.y4 = []
        self.y5 = []
        self.interval = freq
        self.title = title
        self.label = label

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y1.append(command[1])
                self.y2.append(command[2])
                self.y3.append(command[3])
                self.y4.append(command[4])
                self.y5.append(command[5])
                self.ax.plot(self.x, self.y1, 'r', label=self.label)
                self.ax2.plot(self.x, self.y2, 'b')
                self.ax3.plot(self.x, self.y3, 'g', label='a_acc')
                self.ax3.plot(self.x, self.y4, 'y', label='e_acc')
                self.ax4.plot(self.x, self.y5, 'm')
  
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        #self.fig, self.ax = plt.subplots()
        self.fig = plt.figure(figsize=(16,4))
        self.fig.suptitle('{} - {}'.format(self.title, self.label), fontsize=16)

        self.ax = self.fig.add_subplot(1,4,1)
        self.ax.plot([], [], 'r')
        self.ax.set_title('rewards')
        self.ax.set_xlabel('iters')
        self.ax.set_ylabel('')
        #self.ax.legend()

        self.ax2 = self.fig.add_subplot(1,4,2)
        self.ax2.plot([], [], 'b')
        self.ax2.set_title('KL divergence')
        self.ax2.set_xlabel('iters')
        self.ax2.set_ylabel('')

        self.ax3 = self.fig.add_subplot(1,4,3)
        self.ax3.plot([], [], 'g', label='a_acc')
        self.ax3.plot([], [], 'y', label='e_acc')
        self.ax3.set_title("Accuracy")
        self.ax3.set_xlabel('iters')
        self.ax3.set_ylabel('')
        self.ax3.legend()

        self.ax4 = self.fig.add_subplot(1,4,4)
        self.ax4.plot([], [], 'm')
        self.ax4.set_title('measure distance')
        self.ax4.set_xlabel('iters')
        self.ax4.set_ylabel('')

        timer = self.fig.canvas.new_timer(interval=self.interval)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

class Graph:
    def __init__(self, freq, title, label):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(freq=freq, title=title, label=label)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        self.count = 0

    def update(self, reward, entropy, a_acc, e_acc, dist, finished=False):
        self.count += 1

        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send([self.count, reward, entropy, a_acc, e_acc, dist])
