import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np


def error(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    error_low = input[0] - input[1]
    error_high = input[2] - input[0]
    error = []
    error.append(error_low)
    error.append(error_high)
    return error


def average(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    return input[0]


def compare_plot_errorbar(xlabel, ylabel, x, eDivert, woApeX, woRNN, MADDPG):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.errorbar(x=x, y=average(eDivert), yerr=error(eDivert), fmt='r-o', label='e-Divert', capsize=4)
    plt.errorbar(x=x, y=average(woApeX), yerr=error(woApeX), fmt='g-^', label='e-Divert w/o Ape-X', capsize=4)
    plt.errorbar(x=x, y=average(woRNN), yerr=error(woRNN), fmt='m-<', label='e-Divert w/o RNN', capsize=4)
    plt.errorbar(x=x, y=average(MADDPG), yerr=error(MADDPG), fmt='k-*', label='MADDPG', capsize=4)

    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()


def compare_plot(xlabel, ylabel, x, yrange, ours, impala, tsp, ran, eDivert, acktr):
    if os.path.exists('./pdf_xgf') is False:
        os.makedirs('./pdf_xgf')
    pdf = PdfPages('./pdf_xgf/%s-%s.pdf' % (xlabel, ylabel))
    plt.figure(figsize=(13, 13))

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.plot(x, ours, color='b', marker='o', label='DRL-MTMCS', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, impala, color='g', marker='^', label='IMPALA', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, tsp, color='k', marker='d', label='GA-based route planning', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, ran, color='orange', marker='s', label='Random', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, eDivert, color='m', marker='v', label='e-Divert', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, acktr, color='darkred', marker='*', label='ACKTR', markersize=35, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    # if ylabel == "Energy usage (# of full batteries)":
    #     if xlabel == "No. of vehicles":
    #         plt.plot(x, [3.62, 4.62, 5.62, 6.62, 7.62], color='red', linestyle='--', label="Maximum used energy",
    #                  linewidth=4)
    #     else:
    #         plt.axhline(y=4.62, color='red', linestyle='--', label="Maximum used energy", linewidth=4)
    plt.xticks(x, x)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1] * 1.5)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='upper center', fontsize=25, ncol=2, markerscale=0.9)
    plt.tight_layout()

    pdf.savefig()
    plt.close()
    pdf.close()


if __name__ == '__main__':
    # collection_fill
    # compare_plot(xlabel="No. of tasks",
    #              ylabel="Data collection ratio",
    #              x=[1, 5, 10, 15, 20],
    #              yrange=[0, 0.8],
    #              ours=[0.9786, 0.9797, 0.9834, 0.9827, 0.9829],
    #              impala=[0, 0, 0, 0, 0],
    #              tsp=[0.1346, 0.2896, 0.4506, 0.3392, 0.4043],
    #              ran=[0.5564, 0.5706, 0.5566, 0.5287, 0.5161],
    #              )

    # fairness_fill
    # compare_plot(xlabel="No. of tasks",
    #              ylabel="Geographical fairness",
    #              x=[1, 5, 10, 15, 20],
    #              yrange=[0, 0.8],
    #              ours=[0.9806, 0.9808, 0.9818, 0.9831, 0.9825],
    #              impala=[0, 0, 0, 0, 0],
    #              tsp=[0.1741, 0.3742, 0.5102, 0.4178, 0.4684],
    #              ran=[0.5984, 0.5971, 0.5846, 0.5548, 0.5400],
    #              )

    # energy_fill
    # compare_plot(xlabel="No. of tasks",
    #              ylabel="Energy usage (# of full batteries)",
    #              x=[1, 5, 10, 15, 20],
    #              yrange=[0., 4],
    #              ours=[5.0554, 3.4271, 3.1873, 3.0086, 3.0267],
    #              impala=[0, 0, 0, 0, 0],
    #              tsp=[2.2178, 2.2217, 2.5819, 2.3228, 2.4215],
    #              ran=[3.6739, 2.9384, 2.8424, 2.7525, 2.7244],
    #              )
    #
    # efficiency_fill
    # compare_plot(xlabel="No. of tasks",
    #              ylabel="Energy efficiency",
    #              x=[1, 5, 10, 15, 20],
    #              yrange=[0, 0.24],
    #              ours=[0.1545, 0.2440, 0.2557, 0.2625, 0.2623],
    #              impala=[0, 0, 0, 0, 0],
    #              tsp=[0.0085, 0.0584, 0.0810, 0.0597, 0.0807],
    #              ran=[0.0743, 0.0975, 0.0929, 0.0871, 0.0845],
    #              )

    # collection_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Data collection ratio",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.85],
                 ours=[0.9212, 0.9834, 0.9847, 0.9850, 0.9857],
                 impala=[0.8746, 0.9114, 0.9237, 0.9312, 0.9348],
                 tsp=[0.1304, 0.3599, 0.6019, 0.5998, 0.6952],
                 ran=[0.3473, 0.5554, 0.6758, 0.7651, 0.8128],
                 eDivert=[0.496394043, 0.626412354, 0.64434021, 0.710009766, 0.716705322],
                 acktr=[0.74158, 0.88308, 0.9055, 0.94074, 0.95332],
                 )

    # fairness_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Geographical fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.85],
                 ours=[0.9481, 0.9818, 0.9848, 0.9854, 0.9899],
                 impala=[0.8315, 0.9200, 0.9214, 0.9246, 0.9341],
                 tsp=[0.1867, 0.4205, 0.6362, 0.6337, 0.7587],
                 ran=[0.3718, 0.5824, 0.6982, 0.7820, 0.8285],
                 eDivert=[0.525404633, 0.680869488, 0.687950694, 0.735040495, 0.761768641],
                 acktr=[0.76626, 0.90356, 0.92092, 0.95214, 0.96296],
                 )
    # #
    # energy_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Energy usage (# of full batteries)",
                 x=[1, 2, 3, 4, 5],
                 yrange=[1, 5.5],
                 ours=[2.2571, 3.1873, 4.2274, 5.1096, 6.0475],
                 impala=[2.6418, 3.5147, 4.5267, 5.9012, 6.8004],
                 tsp=[1.1563, 2.3891, 3.6308, 4.4241, 5.4243],
                 ran=[1.5269, 2.8528, 4.0060, 5.1084, 6.1423],
                 eDivert=[2.51448, 2.9291, 3.3983, 4.3163, 4.6975],
                 acktr=[2.85786, 4.20736, 5.19638, 6.43056, 7.29908],
                 )

    # efficiency_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.28],
                 ours=[0.3450, 0.2557, 0.1856, 0.1508, 0.1201],
                 impala=[0.2614, 0.2016, 0.1517, 0.1254, 0.1057],
                 tsp=[0.0231, 0.0631, 0.0873, 0.0749, 0.0824],
                 ran=[0.0700, 0.0922, 0.0947, 0.0940, 0.0873],
                 eDivert=[0.092737402, 0.122149955, 0.110833247, 0.102356036, 0.094718901],
                 acktr=[0.17184, 0.15864, 0.13058, 0.1126, 0.10198],
                 )

    # collection-range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Data collection ratio",
                 x=[0.2, 0.4, 0.6, 0.8, 1.0],
                 yrange=[0, 0.8],
                 ours=[0.1445, 0.5502, 0.8106, 0.9275, 0.9834],
                 impala=[0.0896, 0.3803, 0.6674, 0.8392, 0.9234],
                 tsp=[0.0916, 0.1597, 0.2264, 0.2954, 0.3599],
                 ran=[0.0858, 0.2594, 0.4190, 0.5026, 0.5554],
                 eDivert=[0.033710938, 0.103969727, 0.223901367, 0.480507813, 0.626412354],
                 acktr=[0.22838, 0.41498, 0.6259, 0.75982, 0.88308],
                 )

    # fairness_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Geographical fairness",
                 x=[0.2, 0.4, 0.6, 0.8, 1.0],
                 yrange=[0, 0.8],
                 ours=[0.1494, 0.5521, 0.8157, 0.9303, 0.9818],
                 impala=[0.1402, 0.4609, 0.7123, 0.8700, 0.9312],
                 tsp=[0.2414, 0.2693, 0.3308, 0.3767, 0.4205],
                 ran=[0.1369, 0.3414, 0.4739, 0.5383, 0.5824],
                 eDivert=[0.030613357, 0.111296121, 0.264840961, 0.52887742, 0.680869488],
                 acktr=[0.22404, 0.4535, 0.66656, 0.79018, 0.90356],
                 )
    # #
    # energy_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Energy usage (# of full batteries)",
                 x=[0.2, 0.4, 0.6, 0.8, 1.0],
                 yrange=[0, 3.3],
                 ours=[1.7165, 1.8542, 2.5625, 2.8980, 3.1873],
                 impala=[1.782, 2.256, 3.073, 3.461, 3.501],
                 tsp=[1.9900, 2.1049, 2.1832, 2.3070, 2.3891],
                 ran=[2.0124, 2.2358, 2.5617, 2.7329, 2.8528],
                 eDivert=[0.24476, 0.72775, 1.29899, 2.4757, 2.9291],
                 acktr=[1.56374, 2.20794, 3.07106, 3.72122, 4.20736],
                 )

    # efficiency_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Energy efficiency",
                 x=[0.2, 0.4, 0.6, 0.8, 1.0],
                 yrange=[0, 0.21],
                 ours=[0.0180, 0.1410, 0.2173, 0.2502, 0.2557],
                 impala=[0.0039, 0.064, 0.135, 0.1752, 0.2004],
                 tsp=[0.0151, 0.0238, 0.0372, 0.0501, 0.0631],
                 ran=[0.0047, 0.0308, 0.0619, 0.0796, 0.0922],
                 eDivert=[0.006208025, 0.019386733, 0.040419603, 0.082128905, 0.122149955],
                 acktr=[0.0281, 0.07058, 0.11392, 0.13524, 0.15864],
                 )

    # collection_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Data collection ratio",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.8],
                 ours=[0.9686, 0.9720, 0.9782, 0.9817, 0.9834],
                 impala=[0.9114, 0.9187, 0.9203, 0.9196, 0.9174],
                 tsp=[0.3387, 0.3599, 0.3599, 0.3599, 0.3599],
                 ran=[0.5117, 0.5298, 0.5453, 0.5450, 0.5554],
                 eDivert=[0.628780518, 0.582722168, 0.558076172, 0.574083252, 0.626412354],
                 acktr=[0.7672, 0.76748, 0.8309, 0.85354, 0.88308],
                 )

    # fairness_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Geographical fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.8],
                 ours=[0.9704, 0.9743, 0.9805, 0.9807, 0.9818],
                 impala=[0.9242, 0.9267, 0.9289, 0.9175, 0.9143],
                 tsp=[0.4085, 0.4205, 0.4205, 0.4205, 0.4205],
                 ran=[0.5376, 0.5544, 0.5725, 0.5693, 0.5824],
                 eDivert=[0.678998286, 0.628733789, 0.599599653, 0.616799674, 0.680869488],
                 acktr=[0.79694, 0.79146, 0.85366, 0.87072, 0.90356],
                 )
    #
    # energy_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Energy usage (# of full batteries)",
                 x=[1, 2, 3, 4, 5],
                 yrange=[1.5, 3.2],
                 ours=[3.0471, 3.0837, 3.0921, 3.1301, 3.1873],
                 impala=[3.4627, 3.2978, 3.613, 3.502, 3.493],
                 tsp=[2.0290, 2.3146, 2.3176, 2.3767, 2.3891],
                 ran=[2.4501, 2.6070, 2.7080, 2.7820, 2.8528],
                 eDivert=[2.5312, 2.5449, 2.53396, 2.7883, 2.9291],
                 acktr=[3.4221, 3.47782, 3.8745, 4.02894, 4.20736],
                 )

    # efficiency_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.24],
                 ours=[0.2537, 0.2549, 0.2570, 0.2557, 0.2557],
                 impala=[0.210, 0.208, 0.205, 0.203, 0.2014],
                 tsp=[0.0716, 0.0658, 0.0655, 0.0635, 0.0631],
                 ran=[0.0904, 0.0911, 0.0933, 0.0908, 0.0922],
                 eDivert=[0.143993976, 0.123081826, 0.116937443, 0.112135389, 0.122149955],
                 acktr=[0.14434, 0.14264, 0.15442, 0.15516, 0.15864],
                 )
