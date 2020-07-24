import time
import os
import pickle

class Logger:
    def __init__(self, arg_parser):
        save_name = arg_parser.parse_string('save_name')
        now = time.localtime()

        save_name = '.'
        if not os.path.isdir(save_name+'/log'):
            os.makedirs(save_name+'/log')
        if not os.path.isdir(save_name+'/log2'):
            os.makedirs(save_name+'/log2')

        self.log_name = save_name+"/log/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        self.log_name2 = save_name+"/log2/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        self.log = []
        self.log2 = []


    def write(self, num, data):
        if num == 1:
            self.log.append(data)
        if num == 2:
            self.log2.append(data)


    def save(self):
        with open(self.log_name, 'wb') as f:
            pickle.dump(self.log, f)
        with open(self.log_name2, 'wb') as f:
            pickle.dump(self.log2, f)
