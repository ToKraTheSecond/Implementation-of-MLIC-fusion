import cv2
import numpy as np
import scipy.ndimage

from collections import namedtuple

class Bilateral_fusion_MLIC:
    def __init__(self, image_set, kernel_size, scale_depth, alpha, beta):
        self.image_set = image_set
        self.kernel_size = kernel_size
        self.scale_depth = scale_depth
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.001
        self.eta = 1.0
        I_detail_lambda = namedtuple('I_detail_lambda', 'low mid high')
        self.i_detail_lambda = I_detail_lambda(0.95, 0.8, 0.75)
        self.converted_image_set = []
        self.decomposed_image_set = []
        self.difference_set = []
        self.i_detail_set = []
        self.i_detail_d_set = []
        self.i_detail_c_set = []
        self.i_detail_u_set = []
        self.range_gaussian = 0
        self.spatial_gaussian = 0

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
        self.decomposed_image_set.append(image)
        
        if self.scale_depth > 1:
            for scale_step in range(1, self.scale_depth):
                # TODO: Only first try, should be further evaluated
                self.spatial_gaussian = 2 ** (scale_step - 1)
                self.range_gaussian = 0.1 / (2 ** (scale_step - 1))
            
                self.decomposed_image_set.append(self.apply_decomposition_step(image))
    
    def construct_I_detail_set(self):
        self.difference_set = [i_moved - i for i_moved, i in zip(self.decomposed_image_set[1:], self.decomposed_image_set)]

        # TODO: cana we do this in parallel without so many for loops?
        # construct i_detail_d_set <give me my own method pls ;(>
        for image in self.difference_set:
            d_1 = np.sign(image)
            
            # evenly divide into 3 ranges, each raised with different lambda exponent
            d_2 = abs(image)
            d_2_min, d_2_max = d_2.min(), d_2.max()             
            limit_1 = d_2_min + (d_2_max - d_2_min) / 3
            limit_2 = d_2_min + 2 * ((d_2_max - d_2_min) / 3)            
            d_2 = np.where(d_2 < limit_1, abs(d_2) ** self.i_detail_lambda.low, d_2)
            d_2 = np.where(limit_1 < d_2 <= limit_2, abs(d_2) ** self.i_detail_lambda.mid, d_2)
            d_2 = np.where(limit_2 < d_2, abs(d_2) ** self.i_detail_lambda.high, d_2)

            self.i_detail_d_set.append(d_1 * d_2)
        
        # construct i_detail_c_set <give me my own method pls ;(>
        for image in self.decomposed_image_set[1:]:
            gradient_magnitude = self.get_gradient_magnitude(image)
            # TODO: Do we need this dependency?
            # TODO: Check optional params
            image_neighbourhood_min = scipy.ndimage.filters.minimum_filter(image, 3)
            image_neighbourhood_min_with_no_zeros = image_neighbourhood_min + self.epsilon

            self.i_detail_c_set.append(gradient_magnitude / image_neighbourhood_min_with_no_zeros)

        # construct i_detail_u_set <give me my own method pls ;(>
        for d_array, c_array in zip(self.i_detail_d_set, self.i_detail_c_set):
            d_array_abs = np.absolute(d_array)
            # TODO: Global parameters?
            sigma = 8
            kernel_size = 3
            u_array = cv2.GaussianBlur((d_array_abs - c_array), kernel_size, sigma)

            self.i_detail_u_set.append(u_array)

        # construct i_detail_set <gie me my own method pls ;(>
        for u_array, d_array in zip(self.i_detail_u_set, self.i_detail_d_set):
            i_detail = (u_array * d_array) / u_array

            self.i_detail_set.append(i_detail)

    def construct_i_base_robust_maximum(self):
        stacked_set = np.stack(self.decomposed_image_set, axis=0)
        i_base = np.apply_along_axis(func1d=self.robust_maximum, axis=0, arr=stacked_set)

        return i_base
   
    def get_gradient_magnitude(self, image):
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
        
        dx_abs = cv2.convertScaleAbs(dx)
        dy_abs = cv2.convertScaleAbs(dy)

        magnitude = cv2.addWeighted(dx_abs, 0.5, dy_abs, 0.5, 0)

        return magnitude

    def robust_maximum(self, array_1d):    
        indexes = np.argpartition(array_1d, -2)[-2:]
        b2, b1 = np.sort(np.array([array_1d[indexes[0]], array_1d[indexes[1]]]))
        weight = 1
        r = weight * (b1 / b2)
        robust_maximum = ( b1 + b2 * r) / (1 + r)
        
        return robust_maximum

    def fuse(self):
        self.convert_color_space('RGB2YUV')
        self.log_y_channel_set = self.log_y_channels()
        # construct decomposed image set
        [self.apply_decomposition(image) for image in self.log_y_channel_set]

        return self.decomposed_image_set