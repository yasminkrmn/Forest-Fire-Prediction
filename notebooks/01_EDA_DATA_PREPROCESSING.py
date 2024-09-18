#################################################################
# Forest Fires Prediction - EDA
#################################################################

#---------------------------- Dataset ---------------------------

# Attribute Information:
# Date : (DD/MM/YYYY) Day, month (‘june’ to ‘september’), year (2012) Weather data observations
# Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
# RH : Relative Humidity in %: 21 to 90
# Ws :Wind speed in km/h: 6 to 29
# Rain: total day in mm: 0 to 16.8 FWI Components
# Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# Drought Code (DC) index from the FWI system: 7 to 220.4
# Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# Buildup Index (BUI) index from the FWI system: 1.1 to 68
# Fire Weather Index (FWI) Index: 0 to 31.1
# Classes: two classes, namely Fire and not Fire

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#---------------------------- 1. Importing Dataset ---------------------------

df = pd.read_csv("forest_fires_cleaned_dataset.csv")

df.head()
df.info()
df.shape
df.describe()
df_copy=df.drop(['day','month','year'],axis=1)
df_copy.columns

#---------------------------- 2. EDA ---------------------------
df_copy['Classes'].value_counts()
df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)

# Plotting Histogram
df_copy.hist(bins=50, figsize=(15,15))
plt.show()

# Plotting Pie Chart
perc = df_copy['Classes'].value_counts(normalize=True) * 100
classlabels = ["Fire", "Not Fire"]
plt.figure(figsize=(15,15))
plt.pie(perc, labels=classlabels, autopct='%1.1f%%')
plt.title("Pie Chart of Classes")
plt.show()

# Correlation
sns.heatmap(df_copy.corr(),annot=True)
plt.title("Correlation")
plt.show()

# df_copy.to_csv(dataset_path, index=False)


