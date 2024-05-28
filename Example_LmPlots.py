
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from lmplots import lmPlots

#read the example data
data = pd.read_csv('./data/marketing.csv')

#Make a scatter plot

plt.style.use('seaborn-v0_8')
sns.scatterplot(x='youtube', y='sales', data=data)
plt.xlabel("Youtube Ad" )
plt.ylabel("Sales")
plt.show()

#Fit a linear regression model
model = ols("sales ~ youtube", data=data).fit()
print(model.summary())

#make deafult set of plots
lmplot = lmPlots(model)
lmplot()
plt.show()

#Make a single plot
fig, ax = lmplot(which=1)
plt.show()




