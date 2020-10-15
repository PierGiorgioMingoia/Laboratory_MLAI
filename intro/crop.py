import matplotlib.pyplot as plt
import random

cat = plt.imread('img/cat.jpg')
plt.imshow(cat)

h, w = cat.shape[0], cat.shape[1]

side = int(0.5 * min(h, w))
i = random.randint(0, h - side)
j = random.randint(0, w - side)
cat_crop = cat[i:i + side, j:j + side]
plt.imshow(cat_crop)

# grey scale
gray_cat = cat.sum(2)
plt.imshow(gray_cat, cmap='gray')

plt.show()
