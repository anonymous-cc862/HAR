import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

res = pd.read_csv('dia_val.txt', sep='\t').loc[5:, :]
plt.plot(res['steps'].tolist(), res['mean'].tolist(), label='Test accuracy')
plt.plot(res['steps'].tolist(), res['val'].tolist(), label='Validation accuracy')
plt.legend()
plt.show()

plt.scatter(res['val'].tolist(), res['mean'].tolist())
plt.xlabel('Validation accuracy')
plt.ylabel('Test accuracy')
plt.show()




