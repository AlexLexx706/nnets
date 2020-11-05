import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import common

img = mpimg.imread('./images/Abyssinian_120.jpg')
print(type(img))
imgplot = plt.imshow(img)
plt.show()
