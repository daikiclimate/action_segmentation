import numpy as np
from sklearn.metrics import f1_score

import utils


class evaluater:
    def __init__(self, classes=11):
        self.classes = classes

    def set_data(self, label, pred):
        self.x = label
        self.y = pred

    def _print_acc(self, r=5):
        self.accuracy = len(self.x[self.x == self.y]) / len(self.x)
        print("total_accuracy:", round(self.accuracy, r))
        if True:
            for c in range(self.classes):
                x = self.x[self.x == c]
                y = self.y[self.x == c]
                accuracy = len(x[x == y]) / len(x)
                print(utils.id_to_label(c), ":", round(accuracy, r))
        return accuracy

    def print_eval(self, mode=None):
        self._print_acc()
        self.return_F1_purpuse_score()

    def return_eval_score(self):
        return self.accuracy

    def return_F1_purpuse_score(self):
        purpuse_classes = [["painting", "battle"], ["moving", "hidden"]]
        for cls in purpuse_classes:
            cls_index = np.array([utils.label_to_id(i) for i in cls])
            c1 = self.x == cls_index[0]
            c2 = self.x == cls_index[1]
            c = c1 | c2
            a = self.x[c]
            # a = self.reset_label(a, cls_index)
            b = self.y[c]
            a, b = self.reset_label(a, b, cls_index)
            print("f1_score", f1_score(a, b))

    def reset_label(self, label, pred, ind):
        s = label[:]
        ind0 = label == ind[0]
        ind1 = label == ind[1]
        s[ind0] = 0
        s[ind1] = 1

        pr = pred[:]
        ind0 = pred == ind[0]
        ind1 = pred == ind[1]
        pr[ind0] = 0
        pr[ind1] = 1
        ind_exp = ind0 | ind1
        pr[~ind_exp] = -1
        pr[(pr == -1) & (s == 0)] = 1
        pr[(pr == -1) & (s == 1)] = 0
        return s, pr
