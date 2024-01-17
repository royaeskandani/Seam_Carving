import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt


def show_image(img: np.ndarray) -> None:
  """
  Shows an image (img) using matplotlib
  """
  if isinstance(img, np.ndarray):
    if img.shape[-1] == 3 or img.shape[-1] == 4:
      plt.imshow(img[..., :3])
    elif img.shape[-1] == 1 or img.shape[-1] > 4:
      plt.imshow(img[:, :], cmap="gray")
    plt.show()


def convolution2D(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """
  Computes the convolution between kernel and image
  """
  img_new: np.ndarray = np.zeros(img.shape)

  edge: int = kernel.shape[0] // 2
  img_expanded: np.ndarray = np.pad(img, edge, mode="edge")

  for x in range(img_new.shape[0]):
    for y in range(img_new.shape[1]):
      img_part: np.ndarray = img_expanded[x: x + kernel.shape[0], y: y + kernel.shape[0]]
      multiply: np.ndarray = np.multiply(kernel, img_part)

      img_new[x][y] = np.sum(multiply)

  return img_new


def magnitude_of_gradients(img_RGB: np.ndarray) -> np.ndarray:
  """
  Computes the magnitude of gradients using x-sobel and y-sobel 2Dconvolution
  """
  img_gray: np.ndarray = img_RGB[..., :3] @ np.array([0.299, 0.587, 0.114])

  filter_sobel_x: np.ndarray = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
  filter_sobel_y: np.ndarray = filter_sobel_x.T

  img_sobel_x: np.ndarray = convolution2D(img_gray, filter_sobel_x)
  img_sobel_y: np.ndarray = convolution2D(img_gray, filter_sobel_y)

  img_gradient: np.ndarray = np.zeros(img_gray.shape)

  for x in range(img_sobel_x.shape[0]):
    for y in range(img_sobel_y.shape[1]):
      img_gradient[x, y] = np.sqrt(img_sobel_x[x, y] ** 2 + img_sobel_y[x, y] ** 2)  # 2-Norm

  return img_gradient


# Playground
if __name__ == "__main__":
    img = mpimage.imread('tower.jpg')
    img_energy = magnitude_of_gradients(img)
    show_image(img_energy)
