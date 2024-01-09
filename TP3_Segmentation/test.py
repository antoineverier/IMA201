from skimage import io
import matplotlib.pyplot as plt

ima=io.imread('cell.tif')

plt.imshow(ima)
plt.show()