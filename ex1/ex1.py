import cv2
import numpy as np
from itertools import pairwise


def fst(t): return t[0]
def snd(t): return t[1]


def frame_generator(video_file):
    # a generator function that yields one frame at a time from a video file
    cap = cv2.VideoCapture(video_file)  # create a video capture object
    ret = True  # return value
    while ret:  # loop until an error occurs or the end of the video is reached
        ret, frame = cap.read()  # read one frame
        if ret:  # if the frame is valid
            yield frame  # yield the frame
    cap.release()  # release the video capture object


def rgb_to_grayscale(frame):
    # a function that converts a rgb frame to a grayscale frame
    # use cv2.cvtColor to convert the color space
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray  # return the grayscale frame


def grayscale_histogram(frame):
    # a function that returns the histogram of a grayscale frame
    # use cv2.calcHist to calculate the histogram
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist = hist.flatten()  # flatten the histogram to a 1D array
    return hist  # return the histogram


def cumsum_histogram(hist):
    # a function that returns the cumulative sum of a histogram
    cumsum = np.cumsum(hist)  # use np.cumsum to calculate the cumulative sum
    return cumsum  # return the cumulative sum


def norm_difference(cumsum1, cumsum2):
    # a function that returns the norm of the difference between two cumulative sums
    diff = cumsum1 - cumsum2  # calculate the difference
    norm = np.linalg.norm(diff)  # use np.linalg.norm to calculate the norm
    return norm  # return the norm


def argmax(lst):
    return fst(max(((i, x) for i, x in enumerate(lst)), key=snd))


def index_to_to_index_pair(frame_index):
    return (frame_index, frame_index + 1)


def pipeline(*lst):

    param = lst[0]

    for x in lst[1:]:
        if type(x) is tuple:
            param = x[0](*x[1:], param)
        else:
            param = x(param)
    return param


def __plot(x):
    import matplotlib.pyplot as plt
    x = list(x)
    plt.plot(x)
    return x


def main(video_path: str, video_type: int):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    return pipeline(video_path, *main.types[video_type])


main.types = {
    1: [
        frame_generator,
        (map, rgb_to_grayscale),
        (map, grayscale_histogram),
        pairwise,
        (map, lambda t: norm_difference(*t)),
        argmax,
        index_to_to_index_pair
    ],
    2: [
        frame_generator,
        (map, rgb_to_grayscale),
        (map, grayscale_histogram),
        (map, cumsum_histogram),
        pairwise,
        (map, lambda t: norm_difference(*t)),
        argmax,
        index_to_to_index_pair
    ],
    3: [
        frame_generator,
        (map, rgb_to_grayscale),
        (map, grayscale_histogram),
        (map, cumsum_histogram),
        pairwise,
        (map, lambda t: norm_difference(*t)),
        __plot,
        argmax,
        index_to_to_index_pair
    ]
}


# def __plot_2_scences(video_path: str, scene1: int, scene2: int):
#     """
#     Plots the two scenes in the same window
#     :param video_path: path to video file
#     :param scene1: last frame index of the first scene
#     :param scene2: first frame index of the second scene
#     :return: None
#     """
#     cap = cv2.VideoCapture(video_path)

#     # get scene1 frame
#     cap.set(1, scene1)
#     ret, frame1 = cap.read()
#     if not ret:
#         raise ValueError("frame1 not found")

#     # get scene2 frame
#     cap.set(1, scene2)
#     ret, frame2 = cap.read()
#     if not ret:
#         raise ValueError("frame2 not found")

#     cap.release()

#     # plot the two frames
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
#     plt.title("Scene 1")
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
#     plt.title("Scene 2")
#     plt.show()


# if __name__ == "__main__":
#     for vp in [
#         "video1_category1.mp4",
#         "video2_category1.mp4",
#         "video3_category2.mp4",
#         "video4_category2.mp4"
#     ]:
#         scene1, scene2 = main(vp, 2)
#         print(f"video: {vp}, scene1: {scene1}, scene2: {scene2}")
#         __plot_2_scences(vp, scene1, scene2)