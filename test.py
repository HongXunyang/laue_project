import matplotlib.pyplot as plt
from packages import Sample, SampleHolder


sample1 = Sample(1, "LBCO", (0, 0))
sample2 = Sample(2, "LBCO", (0, 1))
sample3 = Sample(3, "LBCO", (1, 0))

sample_holder = SampleHolder(grid_size=(5, 5))
sample_holder.add_sample(sample1)
sample_holder.add_sample(sample2)
sample_holder.add_sample(sample3)

sample1.phi_offset = 10
sample2.phi_offset = 45
sample3.phi_offset = 120


ax = sample_holder.visualize()
plt.show()
print("done")
