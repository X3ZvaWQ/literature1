import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans2, generate_data_for_kmeans

D = generate_data_for_kmeans(
    [10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10], mark=False)
D = kmeans2(D, 6)
plt.scatter(D['x'], D['y'], c=D['c'])
plt.show()
