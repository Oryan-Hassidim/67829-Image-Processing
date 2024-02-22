import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


def pos(x): return x if x > 0 else None


def neg(x): return x if x < 0 else None


def provide_corners(image, min_ratio=0.1):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.int32)

    R = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.int32)

    for w_size in [5, 11]:
        offset = w_size//2
        uvs = [(u-offset, v-offset)
               for u in range(w_size) for v in range(w_size)]
        # Compute the response of the detector at each pixel
        R_ = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.int32)
        for u, v in uvs:
            diff = gray[pos(u):neg(u), pos(v):neg(v)] - \
                gray[pos(-u):neg(-u), pos(-v):neg(-v)]
            R_[pos(u):neg(u), pos(v):neg(v)] += diff * diff
        R_ = R_ / w_size**2
        R = np.maximum(R, R_)

    R2 = R * (R > min_ratio * R.max())

    # get loacl maxima
    R_max = np.zeros((R2.shape[0], R2.shape[1]), dtype=np.bool_)

    R_max[1:-1, 1:-1] = R2[1:-1, 1:-1] > R2[:-2, :-2]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[:-2, 1:-1]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[:-2, 2:]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[1:-1, :-2]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[1:-1, 2:]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[2:, :-2]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[2:, 1:-1]
    R_max[1:-1, 1:-1] *= R2[1:-1, 1:-1] > R2[2:, 2:]

    corners = np.argwhere(R_max)
    return corners, R[corners[:, 0], corners[:, 1]]


def rotate_patch(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    ratio = 1.
    rot_mat = cv.getRotationMatrix2D(image_center, angle, ratio)
    result = cv.warpAffine(
        image, rot_mat, (44, 44), flags=cv.INTER_LINEAR)
    return result


def provide_patches(image, corners):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = gray.astype(np.int32)

    grads = np.zeros((corners.shape[0], 2), dtype=np.int16)
    grads[:, 0] = gray[corners[:, 0]+1, corners[:, 1]] - \
        gray[corners[:, 0]-1, corners[:, 1]]
    grads[:, 1] = gray[corners[:, 0], corners[:, 1]+1] - \
        gray[corners[:, 0], corners[:, 1]-1]

    angles = np.arctan2(grads[:, 1], grads[:, 0])
    angles = np.degrees(angles)

    patches = np.zeros((corners.shape[0], 8, 8, 3), dtype=np.float64)
    for i, (x, y) in enumerate(corners):
        if x-22 < 0 or x+22 > image.shape[0] or y-22 < 0 or y+22 > image.shape[1]:
            continue
        patches[i] = cv.pyrDown(cv.pyrDown(rotate_patch(
            image[x-22:x+22, y-22:y+22].astype(np.uint8), -angles[i])[6:-6, 6:-6]))
        patches[i] -= patches[i].mean()
        patches[i] /= patches[i].std()

    return patches


def find_similar_patch_index(patch, patches, threshold=0.6):
    min1 = np.inf
    min2 = np.inf
    min_idx1 = 0
    for i in range(patches.shape[0]):
        dist = np.linalg.norm(patch - patches[i])
        if dist < min1:
            min2 = min1
            min1 = dist
            min_idx1 = i
        elif dist < min2:
            min2 = dist
    if min2 == 0:
        return None
    if min1/min2 < threshold:
        return min_idx1
    return None


def find_matches(patches1, patches2, corners1, corners2):
    matches = []
    for i, patch in enumerate(patches1):
        index = find_similar_patch_index(patch, patches2, 0.8)
        # and np.linalg.norm(corners1[i] - corners2[index]) <= 400:
        if index is not None:
            matches.append((i, index))
    matches = np.array(matches)
    corners1_m = corners1[matches[:, 0]]
    corners2_m = corners2[matches[:, 1]]
    return corners1_m, corners2_m


def flip_xy(corners):
    return np.array([corners[:, 1], corners[:, 0]]).T


def ransac(corners1, corners2, n_iters=1000, threshold=5):
    corners1 = flip_xy(corners1)
    corners2 = flip_xy(corners2)
    best_model = None
    best_inliers = 0
    for _ in range(n_iters):
        idx = np.random.choice(corners1.shape[0], 4, replace=False)
        c1 = corners1[idx].astype(np.float32)
        c2 = corners2[idx].astype(np.float32)
        model = cv.getPerspectiveTransform(c1, c2)
        corners1_t = cv.perspectiveTransform(
            corners1.reshape(-1, 1, 2).astype(np.float32), model)
        dist = np.linalg.norm(corners1_t.reshape(-1, 2) - corners2, axis=1)
        inliers = np.sum(dist < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_model = model
    return best_model, best_inliers


def main(img1_path: str, img2_path: str, output_path: str):

    print(f'Starting on {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # load images
    print('Loading images...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    # convert to RGB
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    # blur images
    blur1 = cv.GaussianBlur(img1, (5, 5), 0)
    blur2 = cv.GaussianBlur(img2, (5, 5), 0)

    # corners
    print('Finding corners...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    corners1, _ = provide_corners(blur1, 0.1)
    corners2, _ = provide_corners(blur2, 0.2)

    # patches
    print('Providing patches...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    patches1 = provide_patches(blur1, corners1)
    patches2 = provide_patches(blur2, corners2)

    # matches
    print('Finding matches...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    corners1_m, corners2_m = find_matches(
        patches1, patches2, corners1, corners2)

    # ransac
    print('Running RANSAC...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    model, inliers = ransac(corners1_m, corners2_m, 10000)

    # warping image 1
    print('Warping image 1...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    img1_warped = cv.warpPerspective(
        img1, model, (img2.shape[1], img2.shape[0]))

    # load mask
    print('Loading mask...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    alpha = cv.imread(img1_path, cv.IMREAD_UNCHANGED)[:, :, -1]
    mask = cv.warpPerspective(
        alpha, model, (img1.shape[1], img1.shape[0])) > 240
    # merge images
    print('Merging images...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    result = img2.copy()
    result[mask] = img1_warped[mask]

    # save result
    print('Saving result...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    result_bgr = cv.cvtColor(result, cv.COLOR_RGB2BGR)
    cv.imwrite(output_path, result_bgr)

    # plot result
    print('Plotting result...')
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    plt.imshow(result)
    plt.title('Result')
    plt.show()

    print(f'Finished on {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    # main('input/desert_high_res.png', 'input/desert_low_res.jpg', 'desert_output.jpg')
    main('input/lake_high_res.png', 'input/lake_low_res.jpg', 'lake_output.jpg')