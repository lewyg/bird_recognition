class BoundingBox:
    def __init__(self, name, x, y, width, height):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return '<BoundingBox {} ({}, {}) ({}, {})>'.format(self.name, self.x, self.y, self.width, self.height)


class BoundingBoxCollection:
    def __init__(self):
        self.bounding_boxes = list()

    def load(self, filename):
        with open(filename) as file:
            self.bounding_boxes = [self.__read_bounding_box(line) for line in file.readlines()]

    def get(self, name):
        return next(filter(lambda bbox: bbox.name == name, self.bounding_boxes), None)

    def __read_bounding_box(self, line):
        name, *position = line.split()
        x, y, width, height = list(map(int, position))
        name = name.replace('-', '')

        return BoundingBox(name=name, x=x, y=y, width=width, height=height)
