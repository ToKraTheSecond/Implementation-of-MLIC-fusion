import cv2
import numpy as np

class Bilateral_fusion_MLIC:
    def __init__(self, image_set, kernel_size, scale_depth, alfa, beta):
        self.image_set = image_set
        self.kernel_size = kernel_size
        self.scale_depth = scale_depth
        self.alfa = alfa
        self.beta = beta

    def convert_color_space(self, target_space):
        if target_space == 'RGB2YUV':
            converted_image_set = [cv2.cvtColor(image, cv2.COLOR_RGB2YUV) for image in self.image_set]
        elif target_space == 'YUV2RGB':
            converted_image_set = [cv2.cvtColor(image, cv2.COLOR_YUV2RGB) for image in self.image_set]

        return converted_image_set
  
    def get_log_y_channel_set(y_channel_set):
        log_y_channel_set = [np.log(y_channel) for y_channel in y_channel_set]
        
        return log_y_channel_set

    def get_intensity_range(y_channel):
        """Needed for setting range gaussian value."""
        # TODO:
        # Research how to do this
        # min max diff?

        pass

    def apply_decomposition_step(self, image, spatial_gaussian, range_gaussian):
        rows_count, columns_count = image.shape
        output_image = np.zeros((rows_count, columns_count))

        # so we do not need to manually iterate over each pixel in kernel area
        xx, yy = np.meshgrid(range(-self.kernel_size, self.kernel_size + 1), 
                             range(-self.kernel_size, self.kernel_size + 1))

        spatial_response = np.exp( - (xx ** 2 + yy ** 2) / (spatial_gaussian ** 2))

        for row in rows:
            for column in columns:
                # stay inside input image indexing space
                row_min = max(row - self.kernel_size, 1)
                row_max = min(row + self.kernel_size, rows_count)
                column_min = max(column - self.kernel_size, 1)
                column_max = min(column + self.kernel_size, columns_count)
                roi = image[row_min:row_max + 1, column_min, column_max + 1]

                range_response = np.exp(-((roi - image[row, column]) ** 2 / (range_gaussian ** 2))

                output_image[row, column] = (1 / np.sum(responses_product)) * (spatial_response * range_response * roi)
       
        return output_image
    
    def apply_decomposition(self, image):
        """This must be done for every image in input image set."""
        decomposed_images = []
        decomposed_images.append(image)
        
        # TODO: Move it to list comprehension
        for scale_step in range(1, self.scale_depth + 1):
            # TODO: Only first try, should be further evaluated
            spatial_gaussian = 2 ** (scale_step - 1)
            range_gaussian = 0.1 / (2 ** (scale_stel - 1))
            
            decomposed_images.append(apply_decomposition_step(self, image, spatial_gaussian, range_gaussian))

        return decomposed_images