import numpy as np
from numpy.linalg import svd as singular_value_decomposition
from numpy import zeros as create_zeros_matrix
from numpy import matmul as matrix_multiplication
from PIL import Image


# ------------------Class for Image Compression------------------
class ImageCompressor:
    def __init__(self, file_path, singular_value_limit):
        self.file_path = file_path
        self.singular_value_limit = singular_value_limit

    def load_image(self):
        image = Image.open(self.file_path)
        image_channels = np.array(image)
        r = image_channels[:, :, 0]
        g = image_channels[:, :, 1]
        b = image_channels[:, :, 2]
        return r, g, b


# ---------Function for compressing each color channels----------
    @staticmethod
    def compress_channel(channel, limit):
        u, s, v = singular_value_decomposition(channel)
        compressed_channel = create_zeros_matrix((channel.shape[0], channel.shape[1]))
        n = limit

        left_matrix = matrix_multiplication(u[:, 0:n], np.diag(s)[0:n, 0:n])
        inner_compressed = matrix_multiplication(left_matrix, v[0:n, :])
        compressed_channel = inner_compressed.astype('uint8')
        return compressed_channel

# ----Function for compressing & combining all color channels----
    def compress_and_combine_channels(self):
        red_channel, green_channel, blue_channel = self.load_image()

        compressed_red = self.compress_channel(red_channel, self.singular_value_limit)
        compressed_blue = self.compress_channel(blue_channel, self.singular_value_limit)
        compressed_green = self.compress_channel(green_channel, self.singular_value_limit)

        red_array = Image.fromarray(compressed_red)
        blue_array = Image.fromarray(compressed_blue)
        green_array = Image.fromarray(compressed_green)

        compressed_image = Image.merge("RGB", (red_array, green_array, blue_array))
        compressed_image.show()
        compressed_image.save("./results/compressed_robot.jpg")


# ----------------------Image Compression------------------------
if __name__ == "__main__":
    file_path = "data/robot.jpg"
    singular_value_limit = 600

    compressor = ImageCompressor(file_path, singular_value_limit)
    compressor.compress_and_combine_channels()