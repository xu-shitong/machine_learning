import os
import pandas
import matplotlib.pyplot as plt

PATH = os.path.join('')

housing = pandas.read_csv(PATH + 'housing.csv')
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1, s=housing['population']/100, c='population', cmap=plt.get_cmap('jet'))