import math


def L2(x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def L1(x, y):
    return sum([(a - b) for a, b in zip(x, y)])


def sortKey(e):
    return e[0]


def normalize(data, list):
    min_v = min(data)
    max_v = max(data)
    for d in data:
        d = (d - min_v) / (max_v - min_v)

    for i in range(len(list)):
        list[i][0] = data[i]


tomato = (6, 4)
beetroot = (6.5, 8)
label_c = ((120 - 100) / (238 - 100), (0.013 - 0.011) / (0.095 - 0.011))
l1_list = []
l2_list = []
norm_arr = []
if __name__ == '__main__':
    try:
        while True:
            c = input('Label of sample: ')
            tx = float(input('Sample x: '))
            ty = float(input('Sample y: '))
            tx = (tx - 100) / (238 - 100)
            ty = (ty - 0.011) / (0.095 - 0.011)
            p = (tx, ty)
            l2 = L2(label_c, p)
            l2_list.append([l2, c])
            norm_arr.append(l2)

    except KeyboardInterrupt:
        print('\n')
        l2_list.sort(key=sortKey)
        print(l2_list)
