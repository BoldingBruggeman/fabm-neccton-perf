from matplotlib import pyplot as plt
import yaml
import numpy as np


with open("compare_fabm2_vs_fabm3.txt") as f:
    fabm3_data = yaml.safe_load(f)

with open("compare_fabm2_vs_fabm3_o3_qxhost.txt") as f:
    fabm3_opt_data = yaml.safe_load(f)

print(fabm3_data)
print(fabm3_opt_data)

def get_percentage_change(ref_data, exp_data):
    perc_changes = {}
    for name in ref_data:
        ref_time = ref_data[name]
        exp_time = exp_data[name]
        perc_change = exp_time / ref_time * 100
        perc_changes[name.upper()] = perc_change
    return perc_changes


fabm3_perc = get_percentage_change(fabm3_data[0], fabm3_data[1])
fabm3_opt_perc = get_percentage_change(fabm3_opt_data[0], fabm3_opt_data[1])


x = np.arange(len(fabm3_perc))
w = 0.4
fig, ax = plt.subplots()
ax.bar(x-w/2, fabm3_perc.values(), width=w, label="FABM3/O2")
ax.bar(x+w/2, fabm3_opt_perc.values(), width=w, label="FABM3/O3/QxHost")
ax.set_xticks(x)
ax.set_xticklabels(fabm3_perc.keys(), rotation=0, ha='center')
ax.legend(loc="lower right")
ax.set_ylabel("Execution time (% of FABM2/O2)")
ax.grid()
fig.savefig("perf.png", dpi=300)
plt.show()