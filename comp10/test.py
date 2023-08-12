import pickle

with open('data/train.pkl', 'rb') as f:
    obj = pickle.load(f)

target = obj[['f_0', 'target']]
non_float_cols = [c for c in obj.dtypes if c != 'float16']
print(target.head(10))
print(f'columns are {obj.columns}')
print(f'non float cols are {non_float_cols}')
print(f'num rows {len(obj)}')
print('done')