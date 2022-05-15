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


def compare_plot(xlabel, ylabel, x, yrange, ours, dppo, edics, dc, greedy):
    if os.path.exists('./pdf-nuo') is False:
        os.makedirs('./pdf-nuo')
    pdf = PdfPages('./pdf-nuo/%s-%s.pdf' % (xlabel, ylabel))
    plt.figure(figsize=(13, 13))

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.plot(x, ours, color='b', marker='o', label='Our method', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, dppo, color='g', marker='^', label='DPPO', markersize=30, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, edics, color='k', marker='d', label='Edics', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, dc, color='orange', marker='s', label='D&C', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)
    plt.plot(x, greedy, color='purple', marker='v', label='Greedy', markersize=30, markeredgewidth=5,
             markerfacecolor='none',
             linewidth=4)

    # if ylabel == "Energy usage (# of full batteries)":
    #     if xlabel == "No. of vehicles":
    #         plt.plot(x, [3.62, 4.62, 5.62, 6.62, 7.62], color='red', linestyle='--', label="Maximum used energy",
    #                  linewidth=4)
    #     else:
    #         plt.axhline(y=4.62, color='red', linestyle='--', label="Maximum used energy", linewidth=4)

    if ylabel == "Overall collected data" and xlabel == "No. of workers":
        plt.axhline(y=148.63, color='red', linestyle='--', label="The total data", linewidth=4)
    if ylabel == "Overall collected data" and xlabel == "No. of PoIs":
        plt.plot(x, [52.68, 98.13, 148.63, 199.95, 256.65], color='red', linestyle='--', label="The total data",
                 linewidth=4)
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
    compare_plot(xlabel="No. of workers",
                 ylabel="Data collection ratio",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.75],
                 ours=[0.770, 0.849, 0.869, 0.8235, 0.85],
                 dppo=[0.6669, 0.808, 0.85, 0.737, 0.768],
                 edics=[0, 0, 0, 0, 0],
                 dc=[0.2534, 0.398, 0.3368, 0.5941, 0.5434],
                 greedy=[0.0456, 0.3163, 0.4771, 0.5202, 0.5151],
                 )

    # efficiency_fill
    compare_plot(xlabel="No. of workers",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.58],
                 ours=[0.5538, 0.4788, 0.678, 0.51, 0.44],
                 dppo=[0.4533, 0.46, 0.52, 0.32, 0.2955],
                 edics=[0, 0, 0, 0, 0],
                 dc=[0.2596, 0.2947, 0.166, 0.3189, 0.2138],
                 greedy=[0.009, 0.2, 0.2856, 0.2492, 0.1968],
                 )

    # energy_fill
    compare_plot(xlabel="No. of workers",
                 ylabel="Coverage fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.75],
                 ours=[0.75, 0.83, 0.853, 0.82, 0.86],
                 dppo=[0.6343, 0.79, 0.85, 0.726, 0.76],
                 edics=[0, 0, 0, 0, 0],
                 dc=[0.2654, 0.396, 0.3623, 0.562, 0.531],
                 greedy=[0.045, 0.3247, 0.469, 0.51, 0.5215],
                 )
    #

    # fairness_fill
    compare_plot(xlabel="No. of workers",
                 ylabel="Overall collected data",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 125],
                 ours=[114.4451, 126.172, 129.159, 122.396, 126.335],
                 dppo=[99.121, 120.093, 126.3355, 109.5403, 114.1478],
                 edics=[0, 0, 0, 0, 0],
                 dc=[37.66284, 59.15474, 50.058584, 88.301083, 80.765542],
                 greedy=[6.777, 47.011, 70.911, 77.317, 76.559],
                 )

    # fairness_uav
    compare_plot(xlabel="No. of PoIs",
                 ylabel="Data collection ratio",
                 x=[100, 200, 300, 400, 500],
                 yrange=[0, 0.8],
                 ours=[0.95, 0.927, 0.8489, 0.777, 0.7596],
                 dppo=[0.849, 0.8255, 0.808, 0.732, 0.675],
                 edics=[0.595, 0.400, 0.5768, 0.5836, 0.4826],
                 dc=[0.4, 0.4342, 0.398, 0.349, 0.2832],
                 greedy=[0.323, 0.2943, 0.3163, 0.3392, 0.179],
                 )

    # efficiency_uav
    compare_plot(xlabel="No. of PoIs",
                 ylabel="Energy efficiency",
                 x=[100, 200, 300, 400, 500],
                 yrange=[0, 0.45],
                 ours=[0.3, 0.452, 0.478, 0.506, 0.6023],
                 dppo=[0.268, 0.392, 0.46, 0.453, 0.485],
                 edics=[0.13, 0.2, 0.25, 0.283, 0.2165],
                 dc=[0.097, 0.1049, 0.2947, 0.3137, 0.2787],
                 greedy=[0.0646, 0.2324, 0.2, 0.2974, 0.1278],
                 )
    # #
    # energy_uav
    compare_plot(xlabel="No. of PoIs",
                 ylabel="Coverage fairness",
                 x=[100, 200, 300, 400, 500],
                 yrange=[0, 0.75],
                 ours=[0.926, 0.883, 0.83, 0.761, 0.760],
                 dppo=[0.81, 0.804, 0.79, 0.721, 0.663],
                 edics=[0.57, 0.5668, 0.56, 0.575, 0.477],
                 dc=[0.357, 0.406, 0.396, 0.341, 0.2844],
                 greedy=[0.31, 0.269, 0.3247, 0.3274, 0.1812],
                 )

    # collection_uav
    compare_plot(xlabel="No. of PoIs",
                 ylabel="Overall collected data",
                 x=[100, 200, 300, 400, 500],
                 yrange=[0, 200],
                 ours=[50.046, 90.96651, 126.172, 155.3612, 194.9516],
                 dppo=[44.725, 81.006, 120.093, 146.423, 173.187],
                 edics=[31.3446, 39.252, 85.729, 116.691, 123.859],
                 dc=[21.072, 42.608, 59.154, 69.782, 72.683],
                 greedy=[17.016, 28.879, 47.0116, 67.823, 48.940],
                 )
