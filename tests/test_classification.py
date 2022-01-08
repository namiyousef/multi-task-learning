import unittest
import h5py
import numpy as np
from sklearn.svm import SVC
import inspect
# TODO add code to check for dependencies!
DATA_PATH = '../datasets/data_new/{}/{}'
DSET_SIZE = 500

def _get_names(fname):
    names = fname.split('_')
    return names[-1], names[-2]+'.h5'

def _load_data(path):
    with h5py.File(path, 'r') as file:
        key = list(file.keys())[0]
        elems = np.concatenate([elem.flatten().reshape(-1, 1) for elem in file[key][:DSET_SIZE]], axis=-1).T
    file.close()
    return elems


class TestClassification(unittest.TestCase):

    # TODO tests a bit pedantic... this should have been designed a bit better. However it runs properly there is just unnecessary repetition!
    def test_data_import_images_train(self):
        """Tests that data imported correctly
        """
        elems = _load_data(DATA_PATH.format(*_get_names(inspect.stack()[0][3])))
        print(elems.shape)
        return elems

    def test_data_import_images_test(self):
        """Tests that data imported correctly
        """
        elems = _load_data(DATA_PATH.format(*_get_names(inspect.stack()[0][3])))
        print(elems.shape)
        return elems

    def test_data_import_binary_train(self):
        """Tests that data imported correctly
        """
        elems = _load_data(DATA_PATH.format(*_get_names(inspect.stack()[0][3]))).reshape(-1)
        print(elems.shape)
        return elems

    def test_data_import_binary_test(self):
        """Tests that data imported correctly
        """
        elems = _load_data(DATA_PATH.format(*_get_names(inspect.stack()[0][3]))).reshape(-1)
        print(elems.shape)
        return elems


    def test_get_baseline_accuracy(self):
        """Get's a baseline classification accuracy using SVM
        """
        X_train = self.test_data_import_images_train()
        y_train = self.test_data_import_binary_train()
        X_test = self.test_data_import_images_test()
        y_test = self.test_data_import_binary_test()
        model = SVC()
        model.fit(X_train,y_train)
        print((model.predict(X_test) == y_test).mean())


if __name__ == '__main__':
    unittest.main()