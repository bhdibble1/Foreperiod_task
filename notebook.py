# standard deviation instead of quartiles for boxplot

import pandas as pd
import matplotlib.pyplot as plt
import glob

path = "csv_files/"
all_files = glob.glob(path + "*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    
block = []
i=0    
while i < 1000:
    block.append(i)
    i+=20
    

n=1
plt.figure(figsize = (20,30))
for df in li:
    fb = df[(df['block'] == 'fixed-blank') & (df['rt'] >= 0)]
    ff = df[(df['block'] == 'fixed-filled') & (df['rt'] >= 0)]
    v = df[(df['block'] == 'varied') & (df['rt'] >= 0)]
    plt.subplot(6,3,n)
    plt.hist(fb.rt, bins=block, alpha=0.5, edgecolor="black", color="yellow", label="fixed-blank")
    plt.hist(ff.rt, bins=block, alpha=0.5, edgecolor="black", color="blue", label="fixed-filled")
    plt.hist(v.rt, bins=block, alpha=0.5, edgecolor="black", color="orange", label="varied")
    plt.legend(loc='upper right')
    plt.title("FP_P1_" + str(n))
    plt.xlabel("RT in ms")
    plt.ylabel("Number of Trials")
    n+=1
plt.show()

# scattergram

lst = []

for df in li:
    fb = df[(df['block'] == 'fixed-blank') & (df['rt'] >= 0) & (df['correct'] == True)]
    ff = df[(df['block'] == 'fixed-filled') & (df['rt'] >= 0) & (df['correct'] == True)]
    v = df[(df['block'] == 'varied') & (df['rt'] >= 0) & (df['correct'] == True)]
    lst = [fb.rt.mean(), ff.rt.mean(), v.rt.mean()]
    lts = ['fixed-blank', 'fixed-filled', 'varied']
    plt.scatter(lts, lst)
plt.show
    