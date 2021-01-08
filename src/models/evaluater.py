import utils


class evaluater:
    def __init__(self):
        self.classes = 11

    def set_data(self, x, y):
        self.x = x
        self.y = y

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

    def return_eval_score(self):
        return self.accuracy
