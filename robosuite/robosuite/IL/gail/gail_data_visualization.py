import pickle
import matplotlib.pyplot as plt
import os
import sys

order = -1
if len(sys.argv) > 1:
    order = int(sys.argv[1])
curr_dir = os.getcwd().split('/')[-1].upper()

record_names = os.listdir('./log')
record_names.sort()
record_name = './log/' + record_names[order]
with open(record_name, 'rb') as f:
    records = pickle.load(f)
print(record_name)

#iters = [record[0] for record in records]
iters = [i for i in range(len(records))]
rewards = [record[0] for record in records]
a_accs = [record[1] for record in records]
e_accs = [record[2] for record in records]
#a_accs = [record[6] for record in records]
#e_accs = [record[7] for record in records]
Qs = [record[3] for record in records]
disc_entropys = [record[4] for record in records]
#rewards = records

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

start_index = 0

ax1.plot(iters[start_index:], rewards[start_index:], label=curr_dir)
ax1.set_title('TORCS')
ax1.set_xlabel('iters')
ax1.set_ylabel('rewards')
ax1.legend()

ax2.plot(iters[start_index:], a_accs[start_index:], label='a_acc')
ax2.plot(iters[start_index:], e_accs[start_index:], label='e_acc')
ax2.set_title('TORCS')
ax2.set_xlabel('iters')
ax2.set_ylabel('acc')
ax2.legend()

plt.show()
