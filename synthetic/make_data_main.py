import make_data as md
import numpy as np

import pickle



n = 100000

test_split = 0.2

# Do stupid simple method
n_test = int(n*test_split)
n_train = n - n_test

data_dict = dict()


for datatype in ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']:

    X, y, this_datatypes =  md.generate_data(n=n, datatype=datatype)
       
    labs = np.argmax(y, axis=1)

    mask = np.ones(n, dtype=bool)

    test_idxs = np.random.choice(n, size=n_test, replace=False)

    mask[test_idxs] = 0

    train_idxs = np.arange(n)[mask]

    test_idxs = np.arange(n)[~mask]

    print(len(train_idxs), len(test_idxs))

    train_labels = labs[train_idxs]
    test_labels = labs[test_idxs]

    uu, cc = np.unique(train_labels, return_counts=True)
    print('train', uu, cc, cc/np.sum(cc))
    uu, cc = np.unique(test_labels, return_counts=True)
    print('test', uu, cc, cc/np.sum(cc))



    data_dict[datatype] = dict()
    data_dict[datatype]['X'] = X
    data_dict[datatype]['y'] = y
    data_dict[datatype]['labels'] = labs
    data_dict[datatype]['train_idxs'] = train_idxs
    data_dict[datatype]['test_idxs'] = test_idxs
    data_dict[datatype]['datatypes'] = this_datatypes


print('Saving...')
dbfile = open('synthetic_data.p', 'wb')
pickle.dump(data_dict, dbfile)
dbfile.close()
print('Done.')




