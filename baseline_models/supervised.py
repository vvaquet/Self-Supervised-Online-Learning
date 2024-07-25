import numpy as np

class KnnSupervised:

    def __init__(self, initial_data, initial_labels, k, window_size=None):
        self.k = k
        self.data = list(initial_data)
        self.labels = list(initial_labels)
        self.window_size = window_size

        if self.window_size != None and len(self.data) > self.window_size:
            self.data = self.data[-self.window_size:]
            self.labels = self.labels[-self.window_size:]

    def predict_update(self, datapoint, label):
        # Predict
        distances = np.sum((np.array(self.data) - datapoint)**2, axis=1)
        closest_idcs = np.argsort(distances)[:self.k]
        closest_labels = np.array(self.labels)[closest_idcs]
        cur_labels, cur_label_counts = np.unique(closest_labels, return_counts=True)
        prediction = cur_labels[np.argmax(cur_label_counts)]

        # Update
        self.data.append(datapoint)
        self.labels.append(label)

        if self.window_size != None and len(self.data) > self.window_size:
            self.data.pop(0)
            self.labels.pop(0)

        return prediction