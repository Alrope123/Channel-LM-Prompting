import csv
import statistics

taskA_idx = 13
taskB_idx = 14
optimize_against_A_idx = 9
target_prompt_idx = 12
gamma_idx = 6

continuous_prompt_f1_idx = 5
continuous_prompt_f1_sem_idx = 11

mapped_continuous_prompt_f1_idx = 4
mapped_continuous_prompt_f1_sem_idx = 10


all_rows = []
with open("/Users/danielk/ideaProjects/Channel-LM-Prompting/continuous_prompt_experiments_nov11.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        all_rows.append(row)

def filter(taskA, optimized_against_A, gamma=None, taskB=None, target_prompt_length=None):
    filtered_rows = []
    for row in all_rows:
        if target_prompt_length != None and len(row[target_prompt_idx].split(" ")) != target_prompt_length:
            continue

        if taskB and taskB not in row[taskB_idx]:
            continue

        if taskA and row[taskA_idx] != taskA:
            continue

        if optimized_against_A != None and row[optimize_against_A_idx] != optimized_against_A:
            continue

        if gamma != None and float(row[gamma_idx]) != gamma:
            continue

        filtered_rows.append(row)

    count = len(filtered_rows)

    if count > 0:
        avg_f1 = statistics.mean([float(row[continuous_prompt_f1_idx]) for row in filtered_rows])
        avg_f1_sem = statistics.mean([float(row[continuous_prompt_f1_sem_idx]) for row in filtered_rows])

        mapped_avg_f1 = statistics.mean([float(row[mapped_continuous_prompt_f1_idx]) for row in filtered_rows])
        mapped_avg_f1_sem = statistics.mean([float(row[mapped_continuous_prompt_f1_sem_idx]) for row in filtered_rows])

        return {
            'count': count,
            'avg_f1': avg_f1,
            'avg_f1_sem': avg_f1_sem,
            'mapped_avg_f1': mapped_avg_f1,
            'mapped_avg_f1_sem': mapped_avg_f1_sem
        }
    else:
        return None

all_tasks = ['agnews', 'SST-2', 'trec', 'subj', 'sst-5']

if True:
    ## aggregated scores for tasks
    f1_list = []
    f1_sem_list = []
    for taskA in all_tasks:
        r1 = filter(taskA=taskA, optimized_against_A='false', gamma=0)
        r2 = filter(taskA=taskA, optimized_against_A='false', gamma=0.01, taskB='task')
        r3 = filter(taskA=taskA, optimized_against_A='false', gamma=0.01, taskB='prompt')
        f1_list.append([r1['avg_f1'], r2['avg_f1'], r3['avg_f1']])
        f1_sem_list.append([r1['avg_f1_sem'], r2['avg_f1_sem'], r3['avg_f1_sem']])

        print(f"{taskA} \t "
              f"{r1['avg_f1']} \t {r1['avg_f1_sem']} \t "
              f"{r2['avg_f1']} \t {r2['avg_f1_sem']} \t "
              f"{r3['avg_f1']} \t {r3['avg_f1_sem']} \t "
              f"{r1['mapped_avg_f1']} \t {r1['mapped_avg_f1_sem']} \t "
              f"{r2['mapped_avg_f1']} \t {r2['mapped_avg_f1_sem']} \t "
              f"{r3['mapped_avg_f1']} \t {r3['mapped_avg_f1_sem']} \t ")
        print(f"{taskA} (count) \t "
              f"{r1['count']} \t {r1['count']} \t "
              f"{r2['count']} \t {r2['count']} \t "
              f"{r3['count']} \t {r3['count']} \t "
              f"{r1['count']} \t {r1['count']} \t "
              f"{r2['count']} \t {r2['count']} \t "
              f"{r3['count']} \t {r3['count']} \t ")


    import numpy as np
    import matplotlib.pyplot as plt

    data = np.array(f1_list)
    data_err = np.array(f1_sem_list)
    length = len(data)

    # Set plot parameters
    fig, ax = plt.subplots()
    width = 0.2
    x = np.arange(length)

    ax.bar(x, data[:,0], width, color='blue', label='no projection', yerr=data_err[:, 0])
    ax.bar(x + width, data[:,1], width, color='red', label='projection=other tasks', yerr=data_err[:, 1])
    ax.bar(x + (2 * width), data[:,2], width, color='orange', label='projection=random string', yerr=data_err[:, 2])

    ax.set_ylabel('F1')
    ax.set_ylim(0,1.2)
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(all_tasks)
    ax.set_xlabel('Downstream tasks')
    # ax.set_title('Title')
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    fig.tight_layout()
    plt.show()


if True:
    ## aggregated scores for tasks
    x_labels = []
    f1_list = []
    f1_sem_list = []
    for length in range(2, 45):
        for taskA in all_tasks:
            r1 = filter(taskA, optimized_against_A='false', gamma=0, target_prompt_length=length)
            r2 = filter(taskA, optimized_against_A='false', gamma=0.01, taskB='task', target_prompt_length=length)
            r3 = filter(taskA, optimized_against_A='false', gamma=0.01, taskB='prompt', target_prompt_length=length)
            if not r1 or not r2 or not r3:
                continue

            scores = [r1['avg_f1']]
            sem = [r1['avg_f1_sem']]
            if r2 != None:
                scores.append(r2['avg_f1'])
                sem.append(r2['avg_f1_sem'])
            else:
                scores.append(0.0)
                sem.append(0.0)

            if r3 != None:
                scores.append(r3['avg_f1'])
                sem.append(r3['avg_f1_sem'])
            else:
                scores.append(0.0)
                sem.append(0.0)

            f1_list.append(scores)
            f1_sem_list.append(sem)
            x_labels.append(f'{taskA}_{length}')

            # print(f"{taskA} \t "
            #       f"{r1['avg_f1']} \t {r1['avg_f1_sem']} \t "
            #       f"{r2['avg_f1']} \t {r2['avg_f1_sem']} \t "
            #       f"{r3['avg_f1']} \t {r3['avg_f1_sem']} \t "
            #       f"{r1['mapped_avg_f1']} \t {r1['mapped_avg_f1_sem']} \t "
            #       f"{r2['mapped_avg_f1']} \t {r2['mapped_avg_f1_sem']} \t "
            #       f"{r3['mapped_avg_f1']} \t {r3['mapped_avg_f1_sem']} \t ")
            # print(f"{taskA} (count) \t "
            #       f"{r1['count']} \t {r1['count']} \t "
            #       f"{r2['count']} \t {r2['count']} \t "
            #       f"{r3['count']} \t {r3['count']} \t "
            #       f"{r1['count']} \t {r1['count']} \t "
            #       f"{r2['count']} \t {r2['count']} \t "
            #       f"{r3['count']} \t {r3['count']} \t ")

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    data = np.array(f1_list)
    data_err = np.array(f1_sem_list)
    print(f1_list)
    print(f1_sem_list)
    length = len(data)

    # Set plot parameters
    fig, ax = plt.subplots()
    fig.set_figwidth(40)
    # plt.figure(figsize=(40,10))
    width = 0.2
    x = np.arange(length)

    ax.bar(x, data[:,0], width, color='blue', label='no projection', yerr=data_err[:, 0])
    ax.bar(x + width, data[:,1], width, color='red', label='projection=other tasks', yerr=data_err[:, 1])
    ax.bar(x + (2 * width), data[:,2], width, color='orange', label='projection=random string', yerr=data_err[:, 2])
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)


    ax.set_ylabel('F1')
    ax.set_ylim(0,1)
    ax.set_xticks(x + width + width/2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Downstream tasks')
    # ax.set_title('Title')
    ax.legend()
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

    # fig.tight_layout()
    plt.show()


if False:
    ## accuracy as a function of prompt length
    import matplotlib.pyplot as plt
    import pandas as pd

    for taskA in all_tasks:
        length_values = []
        f1_values = []
        for length in range(1, 50):
            r1 = filter(taskA=taskA, optimized_against_A='false', target_prompt_length=length)
            if r1:
                length_values.append(length)
                f1_values.append(r1['avg_f1'])

        # Data
        df=pd.DataFrame({'prompt length': length_values, 'f1': f1_values })

        # multiple line plots
        plt.plot( 'prompt length', 'f1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
        plt.title(f'{taskA} F1 as a function of promot length (tokens)')

        # show legend
        plt.legend()

        # show graph
        plt.show()
