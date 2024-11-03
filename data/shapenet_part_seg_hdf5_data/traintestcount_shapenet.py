import json
import pandas as pd
import glob
import os
import h5py
class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike',
                 'mug', 'pistol', 'rocket', 'skateboard', 'table']
catergory ={
    'airplane':[0],
    'bag':[0],
    'cap':[0],
    'car':[0],
    'chair':[0],
    'earphone':[0],
    'guitar':[0],
    'knife':[0],
    'lamp':[0],
    'laptop':[0],
    'motorbike':[0],
    'mug':[0],
    'pistol':[0],
    'rocket':[0],
    'skateboard':[0],
    'table':[0],
}
catergory_train =catergory


cnt =0
for i in range(6):
    f = h5py.File(f'hdf5_data/ply_data_train{i}.h5')


    label = f['label'][:].astype('int64')
    print(label.shape)
    for j in range(label.shape[0]):
        catergory_train[class_choices[label[j][0]]][0] += 1
    cnt += label.shape[0]
print(f'cnt:{cnt}')
catergory ={
    'airplane':[0],
    'bag':[0],
    'cap':[0],
    'car':[0],
    'chair':[0],
    'earphone':[0],
    'guitar':[0],
    'knife':[0],
    'lamp':[0],
    'laptop':[0],
    'motorbike':[0],
    'mug':[0],
    'pistol':[0],
    'rocket':[0],
    'skateboard':[0],
    'table':[0],
}
catergory_val =catergory
cnt =0
for i in range(1):
    f = h5py.File(f'hdf5_data/ply_data_val{i}.h5')




    label = f['label'][:].astype('int64')

    for j in range(label.shape[0]):
       catergory_val [class_choices [label[j][0]]][0]+=1
    cnt += label.shape[0]
print(f'cnt:{cnt}')


catergory ={
    'airplane':[0],
    'bag':[0],
    'cap':[0],
    'car':[0],
    'chair':[0],
    'earphone':[0],
    'guitar':[0],
    'knife':[0],
    'lamp':[0],
    'laptop':[0],
    'motorbike':[0],
    'mug':[0],
    'pistol':[0],
    'rocket':[0],
    'skateboard':[0],
    'table':[0],
}

catergory_test =catergory
cnt =0
for i in range(2):
    f = h5py.File(f'hdf5_data/ply_data_test{i}.h5')


    label = f['label'][:].astype('int64')

    for j in range(label.shape[0]):

        catergory_test[class_choices[label[j][0]]][0] += 1
    cnt += label.shape[0]
print(f'cnt:{cnt}')
summary ={'train':catergory_train,'val':catergory_val,'test':catergory_test}

df =pd.DataFrame(summary)
df.to_csv('train_cat_summary_shapenet.csv')