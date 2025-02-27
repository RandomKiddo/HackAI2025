"""
TEST FILE

THIS FILE PLOTS THE RAM USAGE IN GIGABYTES PER STEP IN THE OPTIMIZE TEST FILE TAKING
AVERAGES OVER 3 TRIALS. IT ALSO PLOTS THE BAR GRAPH OF THE TIME TO COMPLETE THE PROCESS.
THIS VERSION CURRENTLY DISPLAYS THE TENSORFLOW DATASET VERSION, BUT THE INITIAL VERSION'S
DATA IS STILL COMMENTED TO DISPLAY, IF DESIRED.
"""

import matplotlib.pyplot as plt

#end = ''
end = '_optv1'

def avg_of_3(x1: list, x2: list, x3: list) -> list:
    """
    Returns element-wise average of three lists of the same length as a list of averages.

    x1 [list]: The first list. Required. <br>
    x2 [list]: The second list. Required. <br>
    x3 [list]: The third list. Required. <br>

    Return [list]: List of the element-wise averages.
    """

    x = []
    for _ in range(len(x1)):
        x.append(
            (x1[_]+x2[_]+x3[_])/3
        )
    return x

"""
# INITIAL
y1 = [0.36, 0.364, 0.366, 0.367, 0.38, 0.381, 14.035, 13.747, 12.356, 9.12]
y2 = [0.36, 0.364, 0.367, 0.368, 0.377, 0.377, 16.402, 15.99, 13.558, 10.0]
y3 = [0.36, 0.365, 0.367, 0.369, 0.378, 0.379, 15.981, 15.354, 14.339, 10.228]
t = [150.27657890319824, 159.73844695091248, 173.05346202850342]
"""

# TF
y1 = [0.364, 0.366, 0.368, 0.379, 0.379, 0.391, 0.391, 0.391, 0.391, 0.391]
y2 = [0.36, 0.365, 0.367, 0.367, 0.368, 0.377, 0.378, 0.387, 0.387, 0.387]
y3 = [0.36, 0.366, 0.368, 0.37, 0.38, 0.38, 0.389, 0.389, 0.39, 0.39]
t = [0.5637860298156738, 0.5700221061706543, 0.7763559818267822]

y_avg = avg_of_3(y1, y2, y3)
t_avg = (sum(t))/len(t)
t.append(t_avg)
x = [_ for _ in range(len(y3))]

plt.plot(x, y1, label='Iteration 1')
plt.plot(x, y2, label='Iteration 2')
plt.plot(x, y3, label='Iteration 3')
plt.plot(x, y_avg, linestyle='dashed', label='Average')
plt.xlabel('Step')
plt.ylabel('RAM Usage (GB)')
plt.title('Data Loading RAM Process Usage, TF Dataset Version')
plt.suptitle('2020 2.3 GHz 4-Core i7 | 32 GB DDR4 3733 MHz')
plt.legend()
plt.savefig(f'initial{end}.png')
plt.show()

x = ['Iteration 1', 'Iteration 2', 'Iteration 3', 'Average']
plt.bar(x, t)
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.title('Time for Data Loading Process, TF Dataset Version')
plt.suptitle('2020 2.3 GHz 4-Core i7 | 32 GB DDR4 3733 MHz')
plt.savefig(f'initial_t{end}.png')
plt.show()