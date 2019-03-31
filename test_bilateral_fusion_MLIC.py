import pytest
import cv2

from bilateral_fusion_MLIC import Bilateral_fusion_MLIC

def test_Bilateral_fusion_MLIC_can_be_instantiated():
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=None, kernel_size=None, scale_depth=None, alpha=None, beta=None)

def test_fuse_method_works_with_ok_input():
    test_image = cv2.imread('test_images/lena.png', 1)
    test_image_set = [test_image]
    
    MLIC_fusion = Bilateral_fusion_MLIC(image_set=test_image_set, kernel_size=5, scale_depth=2, alpha=None, beta=None)