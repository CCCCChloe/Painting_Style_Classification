import pandas as pd

f = pd.read_csv("./wikiart_csv/style_train.csv")

#print(f.loc[f['index'] == 12])

small_train_df = pd.DataFrame(columns = ["fname", "label"])

for i in [3,4,9,12,20,21,23,24]:
    small_train_df = small_train_df.append(f.loc[f['label'] == i], ignore_index=True)

small_train = small_train_df.to_csv('small_train.csv')


f = pd.read_csv("./wikiart_csv/style_val.csv")

#print(f.loc[f['index'] == 12])

small_val_df = pd.DataFrame(columns = ["fname", "label"])

for i in [3,4,9,12,20,21,23,24]:
    small_val_df = small_val_df.append(f.loc[f['label'] == i], ignore_index=True)

small_val = small_val_df.to_csv('small_val.csv')