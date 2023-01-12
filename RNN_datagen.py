from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np

class RNNDataGenerator(Sequence):
    def __init__(self, data, labels, batch_size, words):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = words

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Convert to numpy arrays
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)

        return batch_data, to_categorical(batch_labels, num_classes=self.n_classes)
