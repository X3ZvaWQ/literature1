'''
尝试手搓kmeans
'''
import pandas
import numpy
import math

def generate_data_for_kmeans(point_amount_list: list, radius: list, mark: bool) -> pandas.DataFrame:
    D = pandas.DataFrame(columns=['x', 'y', 'c'])
    index = 0
    for i in range(len(point_amount_list)):
        length = point_amount_list[i]
        r = radius[i]
        center = pandas.Series(numpy.random.rand(3)*100, index=['x', 'y', 'c'])
        center['c'] = i if mark else 0
        D.loc[index] = center
        index += 1
        for j in range(length - 1):
            theta = numpy.random.rand() * 2 * math.pi
            _r = numpy.random.rand() * r
            point = pandas.Series(
                [center[0] + _r * math.sin(theta),
                 center[1] + _r * math.cos(theta), i if mark else 0],
                index=['x', 'y', 'c']
            )
            D.loc[index] = point
            index += 1
    return D

def kmeans2(data: pandas.DataFrame, k: int) -> pandas.DataFrame:
    # Initialize centroids
    for i in range(k):
        data.loc[i, 'c'] = i
    while True:
        # Assign each point to its nearest centroid
        for i in range(k, len(data)):
            min_dist = float('inf')
            for j in range(k):
                dist = numpy.linalg.norm(
                    data.loc[i, ['x', 'y']] - data.loc[j, ['x', 'y']])
                if dist < min_dist:
                    min_dist = dist
                    data.loc[i, 'c'] = j
        # Recompute centroids
        centroids = pandas.DataFrame(columns=['x', 'y', 'c'])
        for i in range(k):
            centroids.loc[i] = data[data['c'] == i].mean()
        # Check if centroids have changed
        if (centroids - data.loc[:k-1, ['x', 'y', 'c']] < 1e-4).all().all():
            break
        else:
            data.loc[:k-1, ['x', 'y', 'c']] = centroids
    return data