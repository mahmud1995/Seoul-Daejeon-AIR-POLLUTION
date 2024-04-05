#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
import pandas as pd
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#open the file
pop=pd.read_csv("Number_of_people_data.csv")
anti_s=pd.read_csv("Seoul-Histamin-data.csv")
anti_d=pd.read_csv("Daejeon-Histamin-data.csv")
ozone_s=pd.read_csv("OZONE_Seoul.csv")
ozone_d=pd.read_csv("OZONE_DAEJEON.csv")
dust_s=pd.read_csv("FINEDUST_Seoul.csv")
dust_d=pd.read_csv("FINEDUST_DAEJEON.csv")

##histamin rate for each city
anti_rate_s=anti_s.loc[0,"2022.02.":"2023.02."].div(pop.loc[0,"2022.02.":"2023.02."])
df = pd.DataFrame(anti_rate_s)
anti_rate_s_df = df.transpose()

anti_rate_d=anti_s.loc[0,"2022.02.":"2023.02."].div(pop.loc[1,"2022.02.":"2023.02."])
df0 = pd.DataFrame(anti_rate_d)
anti_rate_d_df = df0.transpose()

ozone_s = ozone_s.loc[0,"2022.02.":"2023.02."]
df1 = pd.DataFrame(ozone_s)
ozone_s_df = df1.transpose()

ozone_d = ozone_d.loc[0,"2022.02.":"2023.02."]
df2 = pd.DataFrame(ozone_d)
ozone_d_df = df2.transpose()

dust_s = dust_s.loc[0,"2022.02.":"2023.02."]
df3 = pd.DataFrame(dust_s)
dust_s_df = df3.transpose()

dust_d = dust_d.loc[0,"2022.02.":"2023.02."]
df4 = pd.DataFrame(dust_d)
dust_d_df = df4.transpose()

data={'anti_rate_s': anti_rate_s_df,'anti_rate_d': anti_rate_d_df,'ozone_s': ozone_s_df
      ,'ozone_d': ozone_d_df,'dust_s': dust_s_df, 'dust_d': dust_d_df}
pd.concat(data, axis=0)
#making the data available to aproach
hisdata=list(anti_rate_s)
for i in list(anti_rate_d):
    hisdata.append(i)
print(hisdata)
dustdata=list(dust_s)
for i in list(dust_d):
    dustdata.append(i)
print(dustdata)

#linear regression model
X = np.array(hisdata)
y = np.array(dustdata)
w = 0.0
b = 0.0


lr = 0.000001

def predict(x,w,b):
    y_pred = w*x + b
    return y_pred

#setting the mse loss
def mse_loss(y_true):
    mse = np.mean((y_true - y_pred)**2)
    return mse


global y_pred
y_pred = predict(X,w,b)

def train(X, y, w, b, lr, epochs):
    for epoch in range(epochs):
        global y_pred
  
        loss = mse_loss(y)
        grad_w = np.mean((y_pred - y)*X)
        grad_b = np.mean(y_pred - y)
        w = w - lr*grad_w
        b = b - lr*grad_b

        y_pred = predict(X,w,b)

        if epoch % 1000 == 0:
            print("Epoch %d: loss=%.4f, w=%.4f, b=%.4f" % (epoch, loss, w, b))
    
    return w, b

w, b = train(X, y, w, b, lr, epochs=35000)
# Read the data from CSV files
pop = pd.read_csv("Number of people data.csv", index_col="name")
dust_pre = pd.read_csv("dust predict data.csv", index_col="name")


# Create a Tkinter window
window = tk.Tk()
window.title("Antihistamin Estimation")
window.geometry("500x700")


def calculate_antihistamin():
    my_city = city_entry.get()
    

    if my_city in dust_pre.index and my_city in pop.index:
        a = dust_pre.loc[my_city]
        esta_antihist = predict(a, w, b)
        c = float(esta_antihist.values[0])
        bb = list(pop.loc[my_city]) 
        estimated_antihistamin = round(c * bb[0])
        result_label.config(text="The estimated use of Antihistamin for {} is {}".format(my_city, estimated_antihistamin))
    else:
        result_label.config(text="City not found in the data")

# Create input label and entry
city_label = tk.Label(window, text="Enter your city (district):")
city_label.pack()

city_entry = tk.Entry(window)
city_entry.pack()

# Create calculate button
calculate_button = tk.Button(window, text="Calculate",width = 20, height = 5, command=calculate_antihistamin, bg = "purple",fg = "white")
calculate_button.pack()

# Create label for displaying the result
result_label = tk.Label(window, text="")
result_label.pack()

# Start the Tkinter event loop
window.mainloop()


# In[13]:


actual_data = pd.read_csv("Real data1.csv", index_col="name")
print(actual_data)
realantihistaminvalue = list(actual_data['antihistamin'])
print(realantihistaminvalue)
realdustvalue = list(actual_data['dust'])
print(realdustvalue)

predicted_antihistamin = []
for i in realdustvalue:
    pre_antihist = predict(i, w, b)
    pc = float(pre_antihist)
    predicted_antihistamin.append(pc)

predicted_ozone = list(realdustvalue)

actual_antihistamin = realantihistaminvalue
actual_dust = realdustvalue

total_predictions = len(actual_antihistamin)

threshold=70
# Calculate the number of correct predictions
correct_predictions = sum([1 for i in range(total_predictions)if abs(predicted_antihistamin[i] - actual_antihistamin[i]) <= threshold])

# Calculate the accuracy percentage
accuracy_percentage = (correct_predictions / total_predictions) * 100

print("Accuracy Percentage:", accuracy_percentage)


# In[ ]:




