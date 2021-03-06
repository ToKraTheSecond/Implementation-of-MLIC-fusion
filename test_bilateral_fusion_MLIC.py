import pytest
import cv2
import numpy as np

from bilateral_fusion_MLIC import Bilateral_fusion_MLIC

def test_Bilateral_fusion_MLIC_can_be_instantiated():
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=None, kernel_size=None, scale_depth=None, alpha=None, beta=0.8)

def test_fuse_method_works_with_one_ok_input():
    test_image = load_test_image()
    test_image_set = [test_image]
    
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=3, alpha=None, beta=0.8)
    MLIC_fusion.fuse()

def test_fuse_method_works_with_multiple_ok_inputs():
    test_image = load_test_image()
    test_image_set = [test_image, test_image, test_image]

    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=3, alpha=None, beta=0.8)
    MLIC_fusion.fuse()

@pytest.mark.parametrize('scale_depth, expected', [
    (0, 0),
    (1, 2),
    (2, 3),
    (3, 4)
])
def test_decomposed_image_set_contains_correct_number_of_image(scale_depth, expected):
    test_image = load_test_image()
    test_image_set = [test_image]

    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=scale_depth, alpha=None, beta=0.8)
    result_image = MLIC_fusion.fuse()
    
    obtained = len(MLIC_fusion.decomposed_image_set)
    
    assert  obtained == expected

def test_first_image_in_decomposed_image_set_is_unchanged():
    test_image = load_test_image()
    test_image_set = [test_image]

    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=2, alpha=None, beta=0.8)
    result_image = MLIC_fusion.fuse()

    expected = MLIC_fusion.log_y_channel_set[0]
    obtained = MLIC_fusion.decomposed_image_set[0]

    np.testing.assert_array_equal(obtained, expected)


def load_test_image():
    # loads in BGR order!
    test_image = cv2.imread('test_images/lena_roi.png', 1)

    return test_image.astype(np.float32)