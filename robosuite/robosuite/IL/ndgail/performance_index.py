import pickle
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

std_line = 0.95
moving_period = 2000 #100
def smoothing(steps, records):
    iters = []
    smooth = []
    for i in range(len(records)):
        if i==0:
            iters.append(steps[i])
        else:   
            iters.append(iters[i-1]+steps[i])
    for i in range(moving_period, len(records)+1):
        a = np.mean(records[i-moving_period:i])
        smooth.append(a)
    iters = iters[moving_period-1:]
    return iters, smooth

def analysis(iters, smooth):
    max = 0
    reach = 0
    last = 0
    for i in range(len(smooth)):
        a = smooth[i]
        if a>= std_line and reach == 0: reach = iters[i]
        if max < a : max = a
        last = a
    return reach, max, last

gan_data_num = 3
if len(sys.argv) == 2:
    start_idx = int(sys.argv[1]) #18
else:
    start_idx = 1
data_num = 3
records1 = []
records2 = []
gail_dir = '../gail/log2'
ndgail_dir = './log2'
record_names = os.listdir(ndgail_dir)
record_names.sort()
gan_record_names = os.listdir(gail_dir)
for order in range(gan_data_num):
    record_name = gail_dir + '/' + gan_record_names[order]
    with open(record_name, 'rb') as f:
        records1.append(pickle.load(f))
for order in range(start_idx, start_idx+data_num):
    record_name = ndgail_dir + '/' + record_names[-order]#temp[order]]
    print(record_name)
    with open(record_name, 'rb') as f:
        records2.append(pickle.load(f))

expert_score = 3000.0
random_score = 0.0

steps1 = []
steps2 = []
rewards1 = []
rewards2 = []
performance2 = []
disc_reward = []
const_reward = []
for i in range(gan_data_num):
    steps1.append([record[0] for record in records1[i]])
    reward1 = [record[1] for record in records1[i]]
    rewards1.append((np.array(reward1)-random_score)/expert_score)
for i in range(data_num):
    steps2.append([record[0] for record in records2[i]])
    reward2 = [record[1] for record in records2[i]]
    rewards2.append((np.array(reward2)-random_score)/expert_score)
    try:
        temp = [record[2] for record in records2[i]]
        temp = temp[moving_period-1:]
        performance2.append(temp)
    except:
        pass
    '''
    try:
        temp = [record[4] for record in records2[i]]
        _, temp = smoothing(steps2[i], temp)
        disc_reward.append(temp)
    except:
        pass
    try:
        temp = [record[5] for record in records2[i]]
        _, temp = smoothing(steps2[i], temp)
        const_reward.append(temp)
    except:
        pass
    '''

iters1 = []
iters2 = []
for i in range(gan_data_num):
    iter1, reward1 = smoothing(steps1[i],rewards1[i])
    iters1.append(iter1)
    rewards1[i] = reward1
for i in range(data_num):
    iter2, reward2 = smoothing(steps2[i],rewards2[i])
    iters2.append(iter2)
    rewards2[i] = reward2
start = max(max([i[0] for i in iters1]), max([i[0] for i in iters2]))
end = min(min([i[-1] for i in iters1]), min([i[-1] for i in iters2]))
iters = np.linspace(start, end, 100000)
for i in range(gan_data_num):
    rewards1[i] = np.interp(iters, iters1[i], rewards1[i])
for i in range(data_num):
    rewards2[i] = np.interp(iters, iters2[i], rewards2[i])
    if len(performance2) > 0:
        performance2[i] = np.interp(iters, iters2[i], performance2[i])
    '''
    if len(disc_reward) > 0:
        disc_reward[i] = np.interp(iters, iters2[i], disc_reward[i])
    if len(const_reward) > 0:
        const_reward[i] = np.interp(iters, iters2[i], const_reward[i])
    '''

stds1 = np.std(rewards1, axis=0)
rewards1 = np.mean(rewards1, axis=0)
stds2 = np.std(rewards2, axis=0)
rewards2 = np.mean(rewards2, axis=0)
if len(performance2) > 0:
    performance2 = np.mean(performance2, axis=0)
    '''
if len(disc_reward) > 0:
    disc_reward = np.mean(disc_reward, axis=0)/np.max(disc_reward)
if len(const_reward) > 0:
    const_reward = np.mean(const_reward, axis=0)/np.max(const_reward)
'''
result1 = analysis(iters, rewards1)
result2 = analysis(iters, rewards2)

percent_line = [std_line for i in range(len(iters))]
contents = ['reach','max','last']
for i in range(len(contents)):
    if i == 0:
        print("[{}] GAIL : {:.0f} | NDGAIL : {:.0f}".format(contents[i],result1[i],result2[i]))
    else:
        print("[{}] GAIL : {:.3f} | NDGAIL : {:.3f}".format(contents[i],result1[i],result2[i]))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

'''
ax1.plot(iters, rewards1, lw=2, label='GAIL', color='blue')
ax1.plot(iters, rewards2, lw=2, label='NDGAIL', color='red')
ax1.fill_between(iters, rewards1-stds1, rewards1+stds1, facecolor='blue', alpha=0.5)
ax1.fill_between(iters, rewards2-stds2, rewards2+stds2, facecolor='red', alpha=0.5)
'''
ax1.plot(iters, rewards1, lw=2, label='GAIL')
ax1.fill_between(iters, rewards1-stds1, rewards1+stds1, alpha=0.3)
ax1.plot(iters, rewards2, lw=2, label='NDGAIL')
ax1.fill_between(iters, rewards2-stds2, rewards2+stds2, alpha=0.3)
if len(performance2) > 0:
    ax1.plot(iters, performance2, lw=2, color='orange')
'''
if len(disc_reward) > 0:
    ax1.plot(iters, disc_reward, lw=2, color='purple')
if len(const_reward) > 0:
    ax1.plot(iters, const_reward, lw=2, color='cyan')
'''
ax1.plot(iters, percent_line, 'k--', lw=1)
ax1.set_title('TORCS\nsmoothed rewards')
ax1.set_xlabel('steps')
ax1.set_ylabel('rewards sum')
ax1.legend()
ax1.grid()

fig.tight_layout()
plt.savefig('{}.png'.format(start_idx))
plt.show()
