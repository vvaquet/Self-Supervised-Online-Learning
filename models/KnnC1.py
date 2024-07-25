import numpy as np
from sklearn.tree import DecisionTreeClassifier


def get_weighted_distance(dist):
    return 1/(1+dist)

class KnnUnsupervisedC1:

    def __init__(self, initial_data, initial_labels, k, window_size, certainty_threshold=None):

        if certainty_threshold is not None:
            self.threshold = certainty_threshold
            self.use_threshold = True
        else:
            self.use_threshold = False

        self.k = k
        self.window_size = window_size

        self.certainties = []
        self.correct_predictions = []
        self.certainties_correct = []
        self.certainties_false = []

        self.data = [initial_data[0,:]]
        self.labels = [initial_labels[0]]
        for i in range(1,len(initial_labels)):
            prediction, certainty = self.predict_with_certainty(initial_data[i,:])
            if prediction == initial_labels[i]:
                self.certainties_correct.append(certainty)
                self.correct_predictions.append(1)
            else:
                self.certainties_false.append(certainty)
                self.correct_predictions.append(0)
            self.certainties.append(certainty)
            
            self.data.append(initial_data[i,:])
            self.labels.append(initial_labels[i])

            if self.window_size != None and len(self.data) > self.window_size:
                self.data = self.data[-self.window_size:]
                self.labels = self.labels[-self.window_size:]

        if not self.use_threshold:
            y = np.array(self.correct_predictions)
            X = np.array(self.certainties).reshape(-1, 1)
            
            self.decision_fct = DecisionTreeClassifier(max_depth=1)
            self.decision_fct.fit(X, y)   




    def metric(self, prediction, closest_labels, closest_dists):
        certainty_t = 0
        certainty_f = 0

        for i in range(closest_labels.shape[0]):
            if closest_labels[i] == prediction:
                certainty_t += get_weighted_distance(closest_dists[i])
            else:
                certainty_f += get_weighted_distance(closest_dists[i])

        return (certainty_t -  certainty_f) /self.k
    
    def predict_with_certainty(self, datapoint):
        distances = np.sum((np.array(self.data) - datapoint)**2, axis=1)
        closest_idcs = np.argsort(distances)[:self.k]
        closest_labels = np.array(self.labels)[closest_idcs]
        closest_dists = distances[closest_idcs]
        cur_labels, cur_label_counts = np.unique(closest_labels, return_counts=True)
        prediction = cur_labels[np.argmax(cur_label_counts)]

        certainty = self.metric(prediction, closest_labels, closest_dists)
        return prediction, certainty

    def predict_update(self, datapoint, label):
        # Predict
        prediction, certainty = self.predict_with_certainty(datapoint)


        if prediction == label:
            self.certainties_correct.append(certainty)
            self.correct_predictions.append(0)
        else:
            self.correct_predictions.append(1)
            self.certainties_false.append(certainty)
        self.certainties.append(certainty)

        # Update
        update = False
        if self.use_threshold:
            if certainty > self.threshold:
                update = True
        else:
            update = self.decision_fct.predict(np.array(certainty).reshape(1, -1))[0]

        if update:
            self.data.append(datapoint)
            self.labels.append(prediction)

            # Take care of window size
            if self.window_size != None and len(self.labels) > self.window_size:
                self.data.pop(0)
                self.labels.pop(0)

        return prediction, update