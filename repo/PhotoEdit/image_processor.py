from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from typing import Tuple

class ImageProcessor:
    """
    Utility class containing static methods for all image transformations and filters.
    Each method takes a PIL Image, creates a copy, applies the operation, and returns the new image.
    All operations are non-destructive.
    """

    @staticmethod
    def apply_grayscale(img: Image.Image) -> Image.Image:
        """
        Convert image to grayscale.
        :param img: Input PIL Image
        :return: Grayscale PIL Image (RGB mode)
        """
        gray = ImageOps.grayscale(img.copy())
        return gray.convert('RGB')

    @staticmethod
    def apply_sepia(img: Image.Image) -> Image.Image:
        """
        Apply sepia tone filter using RGB matrix transformation.
        :param img: Input PIL Image
        :return: Sepia-toned PIL Image (RGB mode)
        """
        img_copy = img.copy().convert('RGB')
        data = list(img_copy.getdata())
        matrix = [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ]
        new_data = []
        for r, g, b in data:
            new_r = min(255, max(0, int(0.393 * r + 0.769 * g + 0.189 * b)))
            new_g = min(255, max(0, int(0.349 * r + 0.686 * g + 0.168 * b)))
            new_b = min(255, max(0, int(0.272 * r + 0.534 * g + 0.131 * b)))
            new_data.append((new_r, new_g, new_b))
        new_img = Image.new('RGB', img_copy.size)
        new_img.putdata(new_data)
        return new_img

    @staticmethod
    def gaussian_blur(img: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur with fixed radius 2.0.
        :param img: Input PIL Image
        :return: Blurred PIL Image
        """
        return img.copy().filter(ImageFilter.GaussianBlur(2.0))
    @staticmethod
    def sharpen(img: Image.Image) -> Image.Image:
        """
    Apply sharpen filter using UnsharpMask (sigma=1.0, amount=2.0).
    :param img: Input PIL Image
    :return: Sharpened PIL Image
    """
        return img.copy().filter(
            ImageFilter.UnsharpMask(radius=1, percent=200, threshold=0)
        )

    @staticmethod
    def rotate_cw(img: Image.Image) -> Image.Image:
        """
        Rotate image 90 degrees clockwise.
        :param img: Input PIL Image
        :return: Rotated PIL Image
        """
        return img.copy().transpose(Image.Transpose.ROTATE_270)

    @staticmethod
    def rotate_ccw(img: Image.Image) -> Image.Image:
        """
        Rotate image 90 degrees counterclockwise.
        :param img: Input PIL Image
        :return: Rotated PIL Image
        """
        return img.copy().transpose(Image.Transpose.ROTATE_90)

    @staticmethod
    def flip_horizontal(img: Image.Image) -> Image.Image:
        """
        Flip image horizontally (mirror).
        :param img: Input PIL Image
        :return: Flipped PIL Image
        """
        return ImageOps.mirror(img.copy())

    @staticmethod
    def flip_vertical(img: Image.Image) -> Image.Image:
        """
        Flip image vertically.
        :param img: Input PIL Image
        :return: Flipped PIL Image
        """
        return ImageOps.flip(img.copy())

    @staticmethod
    def crop_image(img: Image.Image, coords: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop image using bounding box coordinates (left, upper, right, lower).
        :param img: Input PIL Image
        :param coords: Tuple of (left, upper, right, lower)
        :return: Cropped PIL Image
        """
        return img.copy().crop(coords)

    @staticmethod
    def resize_image(
        img: Image.Image,
        size: Tuple[int, int],
        resample: int = Image.Resampling.LANCZOS
    ) -> Image.Image:
        """
        Resize image to given dimensions using specified resampling filter.
        :param img: Input PIL Image
        :param size: Tuple of (width, height)
        :param resample: Resampling filter (default: LANCZOS)
        :return: Resized PIL Image
        """
        return img.copy().resize(size, resample)

    @staticmethod
    def adjust_brightness(img: Image.Image, brightness: float) -> Image.Image:
        """
        Adjust image brightness using ImageEnhance.
        :param img: Input PIL Image
        :param brightness: Adjustment value (-100 to +100); 0=no change
        :return: Brightness-adjusted PIL Image
        """
        enhancer = ImageEnhance.Brightness(img.copy())
        factor = max(0.0, 1.0 + brightness / 100.0)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(img: Image.Image, contrast: float) -> Image.Image:
        """
        Adjust image contrast using ImageEnhance.
        :param img: Input PIL Image
        :param contrast: Adjustment value (-100 to +100); 0=no change
        :return: Contrast-adjusted PIL Image
        """
        enhancer = ImageEnhance.Contrast(img.copy())
        factor = max(0.0, 1.0 + contrast / 100.0)
        return enhancer.enhance(factor)