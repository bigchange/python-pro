# coding=utf-8

T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]

class node:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point
        self.parent = None
        pass

    def set_left(self, left):
        if left is None:
            pass
        left.parent = self
        self.left = left

    def set_right(self, right):
        if right is None:
            pass
        right.parent = self
        self.right = right


def median(lst):
    m = len(lst) / 2
    return lst[m], m


# d指定按照哪一维度划分
def build_kdtree(data, d):
    data = sorted(data, key=lambda x: x[d])
    p, m = median(data)
    tree = node(p)

    del data[m]

    if m > 0:
        tree.set_left(build_kdtree(data[:m], not d))
    if len(data) > 1:
        tree.set_right(build_kdtree(data[m:], not d))
    return tree


def distance(a, b):
    print a, b
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def search_kdtree(tree, d, target):
    if target[d] < tree.point[d]:
        if tree.left is not None:
            return search_kdtree(tree.left, not d, target)
    else:
        if tree.right is not None:
            return search_kdtree(tree.right, not d, target)

    def update_best(t, best):
        if t is None:
            return
        t = t.point
        d = distance(t, target)
        if d < best[1]:
            best[1] = d
            best[0] = t

    best = [tree.point, 100000.0]
    while tree.parent is not None:
        update_best(tree.parent.left, best)
        update_best(tree.parent.right, best)
        tree = tree.parent
    return best[0]


# test
kd_tree = build_kdtree(T, 0)
print search_kdtree(kd_tree, 0, [9, 4])
