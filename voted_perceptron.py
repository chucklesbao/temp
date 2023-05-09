import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test_noans.csv")

train_df['label'].replace(0, -1, inplace = True) #Make all 0 labels -1

word_count = 0
for word_num in train_df.columns[1:-1]: #Count number of words
    word_count += 1

k = 1
c_k = {1: 0} #Map of all c values
w_k = {1: np.zeros(word_count)} #Map of all w vectors
t = 0
num_rounds = 10

while t < num_rounds: #Train
    for index, row in train_df.iterrows():
        y = row['label']
        x = row.values[1:-1]
        if y * np.dot(w_k[k], x) <= 0:
            w_k[k + 1] = w_k[k] + y * x
            c_k[k + 1] = 1
            k += 1
        else:
            c_k[k] += 1
    t += 1

prediction = [] #Create predictions

for index, row in test_df.iterrows():
    sum = 0
    for i in range(1,k+1):
        sum += c_k[i]*np.sign(np.dot(row[1:], w_k[i]))
    prediction.append([index, int(np.sign(sum))])

prediction_df = pd.DataFrame(prediction, columns = ['id','label'])
prediction_df['label'].replace(-1, 0, inplace = True) #Make all -1 labels 0
prediction_df.to_csv("test_ans.csv", index = False)