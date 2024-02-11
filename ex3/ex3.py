import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import click
from functools import partial


def fst(x): return x[0]
def snd(x): return x[1]


def plot_pyramid(pyramid: list[np.ndarray], title: str) -> None:
    """
    Plot the pyramid of images.

    Args:
        pyramid (list[np.ndarray]): List of images in the pyramid.
        title (str): Title of the plot.

    Returns:
        None
    """
    h = int(np.ceil(len(pyramid)/3))
    for i, img in enumerate(pyramid):
        plt.subplot(h, 3, i+1)
        if i < len(pyramid)-1:
            img = (img + 128).astype(np.uint8)
        plt.imshow(img, cmap=img.shape[-1] == 3 and None or 'gray')
        plt.title(f'Level {i}, size: {img.shape[:2]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.gcf().set_size_inches(16, 3*h)
    plt.show()


def gaussian_generator(size: int) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.

    Args:
        size (int): Size of the kernel.

    Returns:
        np.ndarray: 1D Gaussian kernel.
    """
    s = g = np.array([1, 1])
    while len(g) < size:
        g = np.convolve(g, s)
    g = g.astype(np.float32)
    s = g.sum()
    return g.reshape(1, len(g)) / s


def my_pyrDown(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform pyramid downscaling on the image.

    Args:
        img (np.ndarray): Input image.
        kernel (np.ndarray): Gaussian kernel.

    Returns:
        np.ndarray: Downscaled image.
    """
    img = cv.filter2D(img, -1, kernel,
                      borderType=cv.BORDER_REPLICATE)
    img = cv.filter2D(img, -1, kernel.T,
                      borderType=cv.BORDER_REPLICATE)
    img = img[::2, ::2]
    return img


def my_pyrUp(img: np.ndarray, kernel: np.ndarray, dstsize: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Perform pyramid upscaling on the image.

    Args:
        img (np.ndarray): Input image.
        kernel (np.ndarray): Gaussian kernel.
        dstsize (tuple[int, int], optional): Destination size of the upscaled image. Defaults to (0, 0).

    Returns:
        np.ndarray: Upscaled image.
    """
    if dstsize == (0, 0):
        dstsize = (img.shape[0]*2, img.shape[1]*2)
    if len(img.shape) == 3:
        new_img = np.zeros(
            (dstsize[1], dstsize[0], img.shape[2]), dtype=img.dtype)
    else:
        new_img = np.zeros((dstsize[1], dstsize[0]), dtype=img.dtype)
    new_img[::2, ::2] = img
    factor = 0.5 * new_img.size / img.size
    kernel = kernel * factor
    new_img = cv.filter2D(new_img, -1, kernel,
                          borderType=cv.BORDER_REPLICATE)
    new_img = cv.filter2D(new_img, -1, kernel.T,
                          borderType=cv.BORDER_REPLICATE)
    return new_img


def gaussian_pyramid(img: np.ndarray, levels: int, pyrDown=cv.pyrDown) -> list[np.ndarray]:
    """
    Generate a Gaussian pyramid of the image.

    Args:
        img (np.ndarray): Input image.
        levels (int): Number of levels in the pyramid.
        pyrDown (function, optional): Pyramid downscaling function. Defaults to cv.pyrDown.

    Returns:
        list[np.ndarray]: List of images in the Gaussian pyramid.
    """
    pyr = [img]
    for i in range(levels - 1):
        img = pyrDown(img)
        pyr.append(img)

    return pyr


def laplacian_pyramid(pyr: list[np.ndarray], pyrUp=cv.pyrUp) -> list[np.ndarray]:
    """
    Generate a Laplacian pyramid from the Gaussian pyramid.

    Args:
        pyr (list[np.ndarray]): List of images in the Gaussian pyramid.
        pyrUp (function, optional): Pyramid upscaling function. Defaults to cv.pyrUp.

    Returns:
        list[np.ndarray]: List of images in the Laplacian pyramid.
    """
    lap_pyr = []
    for i in range(len(pyr) - 1):
        h, w = pyr[i].shape[:2]
        up = pyrUp(pyr[i + 1], dstsize=(w, h))
        lap_pyr.append(pyr[i].astype(np.int16) - up)
    lap_pyr.append(pyr[-1])
    return lap_pyr


def reconstruct_laplacian_pyramid(lap_pyr: list[np.ndarray], pyrUp=cv.pyrUp) -> np.ndarray:
    """
    Reconstruct the image from the Laplacian pyramid.

    Args:
        lap_pyr (list[np.ndarray]): List of images in the Laplacian pyramid.
        pyrUp (function, optional): Pyramid upscaling function. Defaults to cv.pyrUp.

    Returns:
        np.ndarray: Reconstructed image.
    """
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 2, -1, -1):
        h, w = lap_pyr[i].shape[:2]
        up = pyrUp(img, dstsize=(w, h))
        img = up + lap_pyr[i]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def blend(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, levels: int, kernel_len=5) -> np.ndarray:
    """
    Blend two images using a mask.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        mask (np.ndarray): Mask image.
        levels (int): Number of levels in the pyramid.
        kernel_len (int, optional): Length of the Gaussian kernel. Defaults to 5.

    Returns:
        np.ndarray: Blended image.
    """
    if img1.shape != img2.shape:
        raise ValueError('Images must have the same dimensions')
    if img1.shape[:2] != mask.shape:
        raise ValueError('Images and mask must have the same dimensions')
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32) / 255
    if len(img1.shape) == 3 and len(mask.shape) == 2:
        mask = cv.merge([mask] * img1.shape[-1])

    kernel = gaussian_generator(kernel_len)
    pyr_down = partial(my_pyrDown, kernel=kernel)
    pyr_up = partial(my_pyrUp, kernel=kernel)

    pyr1 = laplacian_pyramid(gaussian_pyramid(img1, levels, pyr_down), pyr_up)
    pyr2 = laplacian_pyramid(gaussian_pyramid(img2, levels, pyr_down), pyr_up)
    mask_pyr = gaussian_pyramid(mask, levels)
    blended_pyr = [mask * img1 + (1 - mask) * img2
                   for (img1, img2, mask) in zip(pyr1, pyr2, mask_pyr)]

    return reconstruct_laplacian_pyramid(blended_pyr, pyr_up)


def hybrid_image(img1: np.ndarray, img2: np.ndarray, levels: int, kernel_len=3) -> np.ndarray:
    """
    Create a hybrid image by combining low-frequency content from one image and high-frequency content from another image.

    Args:
        img1 (np.ndarray): First input image.
        img2 (np.ndarray): Second input image.
        levels (int): Number of levels in the pyramid.
        kernel_len (int, optional): Length of the Gaussian kernel. Defaults to 3.

    Returns:
        np.ndarray: Hybrid image.
    """
    if img1.shape != img2.shape:
        raise ValueError('Images must have the same dimensions')

    kernel = gaussian_generator(kernel_len)
    pyrDown = partial(my_pyrDown, kernel=kernel)
    pyrUp = partial(my_pyrUp, kernel=kernel)

    pyr1 = laplacian_pyramid(gaussian_pyramid(img1, levels, pyrDown), pyrUp)
    plot_pyramid(pyr1, 'Laplacian Pyramid 1')
    pyr2 = laplacian_pyramid(gaussian_pyramid(img2, levels, pyrDown), pyrUp)
    plot_pyramid(pyr2, 'Laplacian Pyramid 2')
    pyr1[-1] = pyr2[-1]
    plot_pyramid(pyr1, 'Laplacian Pyramid 1 with top level replaced')
    return reconstruct_laplacian_pyramid(pyr1, pyrUp)


__read_path = click.Path(exists=True, dir_okay=False, readable=True)
__write_path = click.Path(dir_okay=False, writable=True)


@click.group()
def cli():
    pass


@cli.command(name='blend')
@click.option('--img1_path', '-f', required=True, type=__read_path, prompt="Enter the path of the first image")
@click.option('--img2_path', '-s', required=True, type=__read_path, prompt="Enter the path of the second image")
@click.option('--mask_path', '-m', required=True, type=__read_path, prompt="Enter the path of the mask")
@click.option('--out_path', '-o', required=True, type=__write_path, prompt="Enter the path of the output image")
def blend_command(img1_path: str,
                  img2_path: str,
                  mask_path: str,
                  out_path: str) -> None:
    """
    Command to blend two images using a mask.

    Args:
        img1_path (str): Path of the first image.
        img2_path (str): Path of the second image.
        mask_path (str): Path of the mask.
        out_path (str): Path of the output image.

    Returns:
        None
    """
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    max = np.min([img1.shape[:2], img2.shape[:2]], axis=0)
    print(max)
    img1 = cv.resize(img1, (max[1], max[0]))
    img2 = cv.resize(img2, (max[1], max[0]))
    mask = cv.resize(mask, (max[1], max[0]))
    out = blend(img1, img2, mask, 5)
    cv.imwrite(out_path, out)


@cli.command(name='hybrid')
@click.option('--img1_path', required=True, type=__read_path, prompt="Enter the path of the first image")
@click.option('--img2_path', required=True, type=__read_path, prompt="Enter the path of the second image")
@click.option('--out_path', required=True, type=__write_path, prompt="Enter the path of the output image")
def hybrid_command(img1_path: str,
                   img2_path: str,
                   out_path: str) -> None:
    """
    Command to create a hybrid image by combining low-frequency content from one image and high-frequency content from another image.

    Args:
        img1_path (str): Path of the first image.
        img2_path (str): Path of the second image.
        out_path (str): Path of the output image.

    Returns:
        None
    """
    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
    out = hybrid_image(img1, img2, 5)
    cv.imwrite(out_path, out)


if __name__ == '__main__':
    cli()
