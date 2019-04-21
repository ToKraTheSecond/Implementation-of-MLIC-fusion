import cv2
import numpy as np

class Bilateral_fusion_MLIC:
    def __init__(self, image_set, kernel_size, scale_depth, alpha, beta):
        self.image_set = image_set
        self.kernel_size = kernel_size
        self.scale_depth = scale_depth
        self.alpha = alpha
        self.beta = beta
        self.converted_image_set = []
        self.decomposed_set = []
        self.range_gaussian = 0
        self.spatial_gaussian = 0
        self.epsilon = 0.001

    def convert_color_space(self, target_space):
        if target_space == 'RGB2YUV':
            self.converted_image_set = [cv2.cvtColor(image, cv2.COLOR_RGB2YUV) for image in self.image_set]
        elif target_space == 'YUV2RGB':
            self.converted_image_set = [cv2.cvtColor(image, cv2.COLOR_YUV2RGB) for image in self.image_set]
  
    def log_y_channels(self):
        #Y channel has index 0 - [Y, U, V]
        log_y_channel_set = [np.log(image[:,:,0]) for image in self.converted_image_set]
            
        return log_y_channel_set

    def get_intensity_range(self, y_channel):
        """Needed for setting range gaussian value."""
        # TODO:
        # Research how to do this
        # min max diff?

        pass

    def apply_decomposition_step(self, image):
        rows_count, columns_count = image.shape
        output_image = np.zeros((rows_count, columns_count))

        # spatial_response won't change during iterations so we can precalculate it
        xx, yy = np.meshgrid(range(-self.kernel_size, self.kernel_size + 1), 
                             range(-self.kernel_size, self.kernel_size + 1))

        spatial_response = np.exp( - (xx ** 2 + yy ** 2) / (self.spatial_gaussian ** 2))

        for row in range(rows_count):
            for column in range(columns_count):
                # stay inside input image indexing space
                row_min = max(row - self.kernel_size, 0)
                row_max = min(row + self.kernel_size, rows_count - 1)
                column_min = max(column - self.kernel_size, 0)
                column_max = min(column + self.kernel_size, columns_count - 1)
                roi = image[row_min:row_max + 1, column_min:column_max + 1]

                # spatial_response offsets -> only correct spatial_response region will be indexed when indexed pixel is near image border
                r_o = -1 * row + self.kernel_size
                c_o = -1 * column + self.kernel_size
                spatial_response_off = spatial_response[row_min + r_o:row_max + r_o + 1,
                                                       column_min + c_o:column_max + c_o + 1]

                range_response = np.exp(-1 * ((roi - image[row, column]) ** 2 / (self.range_gaussian ** 2)))                
                responses_product = spatial_response_off * range_response

                output_image[row, column] = (1 / np.sum(responses_product)) * np.sum((responses_product * roi))
       
        return output_image
    
    def apply_decomposition(self, image):
        """This must be done for every image in input image set."""
        decomposed_images = []
        decomposed_images.append(image)
        
        if self.scale_depth > 1:
            for scale_step in range(1, self.scale_depth):
                # TODO: Only first try, should be further evaluated
                self.spatial_gaussian = 2 ** (scale_step - 1)
                self.range_gaussian = 0.1 / (2 ** (scale_step - 1))
            
                decomposed_images.append(self.apply_decomposition_step(image))

        return decomposed_images

    @staticmethod
    def get_gradient_magnitude(image):
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        
        dx_abs = cv2.convertScaleAbs(dx)
        dy_abs = cv2.convertScaleAbs(dy)

        magnitude = cv2.addWeighted(dx_abs, 0.5, dy_abs, 0.5, 0)

        return magnitude

    def fuse(self):
        self.convert_color_space('RGB2YUV')
        self.log_y_channel_set = self.log_y_channels()
        self.decomposed_set = [self.apply_decomposition(image) for image in self.log_y_channel_set]

        return self.decomposed_set