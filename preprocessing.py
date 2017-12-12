class Indexer:
    def __init__(self):
        self.c2n = dict()
        self.n2c = list()
        self.start_idx = 1

    def fit_transform(self, value, add_if_new=True):
        n = self.c2n.get(value)

        if n is None and add_if_new:
            n = len(self.n2c) + self.start_idx
            self.c2n[value] = n
            self.n2c.append(value)
        return n

    def value(self, number):
        return self.n2c[number]

    def max_number(self):
        return len(self.n2c)

    def transform(self, data, matrix=False, add_if_new=True):
        vectors = []
        for line in data:
            if matrix:
                line_built = []
                for c in line:
                    line_built.append(self.fit_transform(c, add_if_new=True))
                vectors.append(line_built)
            else:
                vectors.append(self.fit_transform(line, add_if_new=True))
        return vectors

def transform(features, cv, idxer, add_if_new=True):
    X = cv.transform(features)
    X = cv.inverse_transform(X)
    X = idxer.transform(X, idxer, add_if_new=add_if_new)
    return X


def read_file(filename, label=False, raw=False):
    with open(filename) as file:
        lines = file.readlines()
    if raw:
        return lines
    elif label:
        return [int(line.strip()) for line in lines]
    else:
        return [list(line.strip()) for line in lines]


# if __name__ == "__main__":
