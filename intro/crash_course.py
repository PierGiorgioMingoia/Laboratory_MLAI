import numpy as np

x = np.random.random((2, 3))
print(x)

a = 5
b = None
if a > 5:
    print("a>5")
elif a == 5:
    print("a=5")
else:
    print("a<5")

for a in range(5):
    print(a)

x = 3
print(type(x))

x = "Hello"
if isinstance(x, str):
    print("This is a string")
else:
    print(type(x))


# Classes
class Book:
    # constructor
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def pretty_print(self):
        print('"' + self.title + '" by ' + self.author)


b = Book("The Lord of The Rings", "J.R.R Tolkien")
b.pretty_print()

# string manipulation
string = "Hello World"
hello = string[:6]
world = string[6:]

# string are immutable
print(hello + world)
print("{}{}".format(hello, world))
print("%s%s %d" % (hello, world, 1))
splitted = string.split(" ")
print('_'.join(splitted))

# list definiton manipulation
my_numbers = [4, 5, 6, 7, 856, 12]
squares = []
even_numbers = []
for el in my_numbers:
    squares.append(el ** 2)
    if el % 2 == 0:
        even_numbers.append(el)
print("Squares: ", squares)
print("Even numbers: ", even_numbers)

# filtering

even_numbers = [el for el in my_numbers if el % 2 == 0]
print(even_numbers)

import math


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


print(euclidean_distance(0, 0, 1, 1))

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("a = \n", a)
print("Shape: ", a.shape)

a = b = [[1, 2], [3, 4]]
c = [[0, 0], [0, 0]]

for i in range(len(a)):
    for j in range(len(b)):
        c[i][j] = a[i][j] * b[i][j]

a = b = np.array([[1, 2, 3], [4, 5, 6]])
c = a * b
# all operations can be done like this
print(c)
