from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

# 构建数据
interval=50
plt.figure(figsize=(8, 12))
x_data = ['Task 1-20', 'Task 1-15', 'Task 1-10','Task 1-5', 'Task 1']
ways = ['DRL-MTVCS', 'IMPALA', 'Random', 'GA approach']
colors = ['green', 'red', 'blue', 'orange']
bar_width_per_task = 2.3
bar_height_per_task=bar_width_per_task*0.8

task1ours = [4, 23]
task2ours = [23]
task3ours = [34]
task4ours = [12]
task5ours = [54]
task6ours = []
task7ours = []
task8ours = []
task9ours = []
task10ours = []
task11ours = []
task12ours = []
task13ours = []
task14ours = []
task15ours = []
task16ours = []
task17ours = []
task18ours = []
task19ours = []
task20ours = []

task1impala = [4, 23]
task2impala = [23]
task3impala = [34]
task4impala = [12]
task5impala = [54]
task6impala = []
task7impala = []
task8impala = []
task9impala = []
task10impala = []
task11impala = []
task12impala = []
task13impala = []
task14impala = []
task15impala = []
task16impala = []
task17impala = []
task18impala = []
task19impala = []
task20impala = []

task1random = [0.5607] * 5
task2random = [0.5928] * 4
task3random = [0.5475] * 4
task4random = [0.6075] * 4
task5random = [0.6338] * 4
task6random = [0.5202] * 3
task7random = [0.5584] * 3
task8random = [0.6194] * 3
task9random = [0.4464] * 3
task10random = [0.4856] * 3
task11random = [0.4366] * 2
task12random = [0.5598] * 2
task13random = [0.4802] * 2
task14random = [0.4689] * 2
task15random = [0.4926] * 2
task16random = [0.5817] * 1
task17random = [0.6066] * 1
task18random = [0.5570] * 1
task19random = [0.3866] * 1
task20random = [0.2517] * 1

task1tsp = [0.0994] * 5
task2tsp = [0.1959] * 4
task3tsp = [0.1174] * 4
task4tsp = [0.2463] * 4
task5tsp = [0.9887] * 4
task6tsp = [0.2958] * 3
task7tsp = [0.3427] * 3
task8tsp = [0.8385] * 3
task9tsp = [0.7697] * 3
task10tsp = [0.1261] * 3
task11tsp = [0.8790] * 2
task12tsp = [0.1492] * 2
task13tsp = [0.1657] * 2
task14tsp = [0.7092] * 2
task15tsp = [0.0855] * 2
task16tsp = [0.6316] * 1
task17tsp = [0.5180] * 1
task18tsp = [0.6405] * 1
task19tsp = [0.1227] * 1
task20tsp = [0.0164] * 1

# plt.barh(y=np.arange(len(task1ours))*interval + bar_width_per_task * 0, width=task1ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task2ours)) *interval+ bar_width_per_task * 1, width=task2ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task3ours))*interval + bar_width_per_task * 2, width=task3ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task4ours)) *interval+ bar_width_per_task * 3, width=task4ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task5ours))*interval + bar_width_per_task * 4, width=task5ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task6ours))*interval + bar_width_per_task * 5, width=task6ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task7ours))*interval + bar_width_per_task * 6, width=task7ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task8ours)) *interval+ bar_width_per_task * 7, width=task8ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task9ours)) *interval+ bar_width_per_task * 8, width=task9ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task10ours)) *interval+ bar_width_per_task * 9, width=task10ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task11ours)) *interval+ bar_width_per_task * 10, width=task11ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task12ours))*interval + bar_width_per_task * 11, width=task12ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task13ours))*interval + bar_width_per_task * 12, width=task13ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task14ours)) *interval+ bar_width_per_task * 13, width=task14ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task15ours))*interval + bar_width_per_task * 14, width=task15ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task16ours)) *interval+ bar_width_per_task * 15, width=task16ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task17ours)) *interval+ bar_width_per_task * 16, width=task17ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task18ours))*interval + bar_width_per_task * 17, width=task18ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task19ours))*interval + bar_width_per_task * 18, width=task19ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
# plt.barh(y=np.arange(len(task20ours)) *interval+ bar_width_per_task * 19, width=task20ours, label=ways[0], alpha=0.8,
#          height=bar_width_per_task, color=colors[0])
#
#
#
# plt.barh(y=np.arange(len(task1impala))*interval + bar_width_per_task * 0, width=task1impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task2impala))*interval + bar_width_per_task * 1, width=task2impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task3impala))*interval + bar_width_per_task * 2, width=task3impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task4impala)) *interval+ bar_width_per_task * 3, width=task4impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task5impala))*interval + bar_width_per_task * 4, width=task5impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task6impala)) *interval+ bar_width_per_task * 5, width=task6impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task7impala)) *interval+ bar_width_per_task * 6, width=task7impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task8impala))*interval + bar_width_per_task * 7, width=task8impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task9impala))*interval + bar_width_per_task * 8, width=task9impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task10impala))*interval + bar_width_per_task * 9, width=task10impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task11impala))*interval + bar_width_per_task * 10, width=task11impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task12impala))*interval + bar_width_per_task * 11, width=task12impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task13impala))*interval + bar_width_per_task * 12, width=task13impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task14impala))*interval + bar_width_per_task * 13, width=task14impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task15impala))*interval + bar_width_per_task * 14, width=task15impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task16impala))*interval + bar_width_per_task * 15, width=task16impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task17impala))*interval + bar_width_per_task * 16, width=task17impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task18impala))*interval + bar_width_per_task * 17, width=task18impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task19impala))*interval + bar_width_per_task * 18, width=task19impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])
# plt.barh(y=np.arange(len(task20impala))*interval + bar_width_per_task * 19, width=task20impala, label=ways[1], alpha=0.8,
#          height=bar_width_per_task, color=colors[1])

plt.barh(y=np.arange(len(task5tsp)) * interval + bar_width_per_task * 4, width=task5tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])

plt.barh(y=np.arange(len(task1random))*interval + bar_width_per_task * 0, width=task1random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task2random))*interval + bar_width_per_task * 1, width=task2random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task3random))*interval + bar_width_per_task * 2, width=task3random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task4random))*interval + bar_width_per_task * 3, width=task4random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task5random))*interval + bar_width_per_task * 4, width=task5random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task6random))*interval + bar_width_per_task * 5, width=task6random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task7random))*interval + bar_width_per_task * 6, width=task7random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task8random))*interval + bar_width_per_task * 7, width=task8random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task9random))*interval + bar_width_per_task * 8, width=task9random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task10random))*interval + bar_width_per_task * 9, width=task10random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task11random))*interval + bar_width_per_task * 10, width=task11random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task12random))*interval + bar_width_per_task * 11, width=task12random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task13random))*interval + bar_width_per_task * 12, width=task13random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task14random))*interval + bar_width_per_task * 13, width=task14random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task15random))*interval + bar_width_per_task * 14, width=task15random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task16random))*interval + bar_width_per_task * 15, width=task16random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task17random))*interval + bar_width_per_task * 16, width=task17random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task18random))*interval + bar_width_per_task * 17, width=task18random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task19random))*interval + bar_width_per_task * 18, width=task19random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])
plt.barh(y=np.arange(len(task20random))*interval + bar_width_per_task * 19, width=task20random, label=ways[2], alpha=1,
         height=bar_height_per_task, color=colors[2])

plt.barh(y=np.arange(len(task1tsp)) * interval + bar_width_per_task * 0, width=task1tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task2tsp)) * interval + bar_width_per_task * 1, width=task2tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task3tsp)) * interval + bar_width_per_task * 2, width=task3tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task4tsp)) * interval + bar_width_per_task * 3, width=task4tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
# plt.barh(y=np.arange(len(task5tsp)) * interval + bar_width_per_task * 4, width=task5tsp, label=ways[3], alpha=1,
#          height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task6tsp)) * interval + bar_width_per_task * 5, width=task6tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task7tsp)) * interval + bar_width_per_task * 6, width=task7tsp, label=ways[3],alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task8tsp)) * interval + bar_width_per_task * 7, width=task8tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task9tsp)) * interval + bar_width_per_task * 8, width=task9tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task10tsp)) * interval + bar_width_per_task * 9, width=task10tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task11tsp)) * interval + bar_width_per_task * 10, width=task11tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task12tsp)) * interval + bar_width_per_task * 11, width=task12tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task13tsp)) * interval + bar_width_per_task * 12, width=task13tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task14tsp)) * interval + bar_width_per_task * 13, width=task14tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task15tsp)) * interval + bar_width_per_task * 14, width=task15tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task16tsp)) * interval + bar_width_per_task * 15, width=task16tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task17tsp)) * interval + bar_width_per_task * 16, width=task17tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task18tsp)) * interval + bar_width_per_task * 17, width=task18tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task19tsp)) * interval + bar_width_per_task * 18, width=task19tsp, label=ways[3],alpha=1,
         height=bar_height_per_task, color=colors[3])
plt.barh(y=np.arange(len(task20tsp)) * interval + bar_width_per_task * 19, width=task20tsp, label=ways[3], alpha=1,
         height=bar_height_per_task, color=colors[3])

# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for y, x in enumerate(y_data):
#     plt.text(x+5000, y-bar_width/2, '%s' % x, ha='center', va='bottom')
# for y, x in enumerate(y_data2):
#     plt.text(x+5000, y+bar_width/2, '%s' % x, ha='center', va='bottom')
# 为Y轴设置刻度值
plt.yticks(np.arange(len(x_data))*interval + [bar_width_per_task * 9.5, bar_width_per_task * 7, bar_width_per_task * 4.5,
                                     bar_width_per_task * 2, 0], x_data)  # todo:[5]
# 设置标题
# 为两条坐标轴设置名称
plt.xlabel("Data collection ratio")
# plt.ylabel("2")
# 显示图例
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.show()
