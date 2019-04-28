import pytest
import cv2
import numpy as np

from bilateral_fusion_MLIC import Bilateral_fusion_MLIC

def test_Bilateral_fusion_MLIC_can_be_instantiated():
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=None, kernel_size=None, scale_depth=None, alpha=None, beta=None)

def test_fuse_method_works_with_ok_input():
    test_image = load_test_image()
    test_image_set = [test_image]
    
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=2, alpha=None, beta=None)
    MLIC_fusion.fuse()

@pytest.mark.parametrize('scale_depth, expected', [
    (1, 1),
    (2, 2),
    (3, 3)
])
def test_i_base_set_contains_correct_number_of_image(scale_depth, expected):
    test_image = load_test_image()
    test_image_set = [test_image]

    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=scale_depth, alpha=None, beta=None)
    MLIC_fusion.fuse()
    
    obtained = len(MLIC_fusion.i_base_set[0])
    
    assert  obtained == expected

def test_first_image_in_i_base_set_is_unchanged():
    test_image = load_test_image()
    test_image_set = [test_image]

    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=3, scale_depth=1, alpha=None, beta=None)
    MLIC_fusion.fuse()

    expected = MLIC_fusion.log_y_channel_set[0]
    obtained = MLIC_fusion.i_base_set[0][0]

    np.testing.assert_array_equal(obtained, expected)


def load_test_image():
    test_image = cv2.imread('test_images/lena_roi.png', 1)

    return test_image