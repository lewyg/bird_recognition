class BoundingBox:
    def __init__(self, name, x1, x2, y1, y2):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def __repr__(self):
        return '<BoundingBox {} ({}, {}) ({}, {})>'.format(self.name, self.x1, self.y1, self.x2, self.y2)


class BoundingBoxCollection:
    def __init__(self):
        self.bounding_boxes = list()

    def load(self, filename):
        with open(filename) as file:
            self.bounding_boxes = [self.__read_bounding_box(line) for line in file.readlines()]

    def get(self, name):
        return next(filter(lambda bbox: bbox.name == name, self.bounding_boxes), None)

    def __read_bounding_box(self, line):
        name, x1, x2, y1, y2 = line.split()
        name = name.replace('-', '')

        return BoundingBox(name=name, x1=min(x1, x2), x2=max(x1, x2), y1=min(y1, y2), y2=max(y1, y2))
