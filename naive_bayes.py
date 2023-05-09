import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test_noans.csv")

train_word_data_0 = {} #Map where we store count of all word data for label 0
train_word_data_1 = {} #Map where we store count of all word data for label 1

count_0 = 0
count_1 = 0
count_0_words = 0
count_1_words = 0

for label in train_df['label']: #Count number of labels
    if label == 0:
        count_0 += 1
    else:
        count_1 += 1

count_total = count_0 + count_1

for word_num in train_df.columns[1:-1]: #Init maps and add 1 smoothing
    train_word_data_0[word_num] = 1
    train_word_data_1[word_num] = 1
    count_0_words += 1
    count_1_words += 1

for index, row in train_df.iterrows(): #Count words
    if row['label'] == 0:
        for word_num, count in row[1:-1].items():
            train_word_data_0[word_num] += count
            count_0_words += count
    else:
        for word_num, count in row[1:-1].items():
            train_word_data_1[word_num] += count
            count_1_words += count
            
prediction = [] #Create predictions

for index, row in test_df.iterrows():
    pred = np.zeros(2)
    for word_num, count in row[1:].items():
        pred[0] += count * np.log(train_word_data_0[word_num]/count_0_words)
        pred[1] += count * np.log(train_word_data_1[word_num]/count_1_words)
    pred[0] += np.log(count_0/count_total)
    pred[1] += np.log(count_1/count_total)
    prediction.append([index, np.argmax(pred)])

prediction_df = pd.DataFrame(prediction, columns = ['id','label'])
prediction_df.to_csv("test_ans.csv", index = False)
    