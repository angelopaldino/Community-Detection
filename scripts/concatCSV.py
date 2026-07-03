import pandas as pd
train = pd.read_csv('C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\TAPE_CoraRAW\\train.csv')
test = pd.read_csv('C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\TAPE_CoraRAW\\test.csv')
val = pd.read_csv('C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\TAPE_CoraRAW\\val.csv')
df = pd.concat([train, test, val], ignore_index=True)
df.to_csv('C:\\Users\\angel\\OneDrive\\Desktop\\Community Detection LLM\\data\\processed\\combined.csv', index=False)
print(f"Uniti {len(df)} righe")
print(df['id'].head(10));
print('min:', df['id'].min(), 'max:', df['id'].max(), 'unici:', df['id'].nunique())

