from typing import Callable
from ex1 import *
import matplotlib.pyplot as plt


def plot(x):
    x = list(x)
    plt.subplot(3, 4, (1,4))
    plt.plot(x)
    return x


main.types[3] = [
    frame_generator,
    (map, bgr_to_grayscale),
    (map, grayscale_histogram),
    pairwise,
    (map, lambda t: norm_difference(*t)),
    plot,
    argmax,
    index_to_to_index_pair
]
main.types[4] = [
    frame_generator,
    (map, bgr_to_grayscale),
    (map, grayscale_histogram),
    (map, cumsum_histogram),
    pairwise,
    (map, lambda t: norm_difference(*t)),
    plot,
    argmax,
    index_to_to_index_pair
]

def plot_2_scences(video_path: str, scene1: int, scene2: int):
    """
    Plots the two scenes in the same window
    :param video_path: path to video file
    :param scene1: last frame index of the first scene
    :param scene2: first frame index of the second scene
    :return: None
    """
    cap = cv2.VideoCapture(video_path)

    # get scene1 frame
    cap.set(1, scene1)
    ret, frame1 = cap.read()
    if not ret:
        raise ValueError("frame1 not found")

    # get scene2 frame
    cap.set(1, scene2)
    ret, frame2 = cap.read()
    if not ret:
        raise ValueError("frame2 not found")

    cap.release()

    # plot the two frames
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.title("Scene 1")
    # remove the axis
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    plt.title("Scene 2")
    plt.axis("off")

def plot_4_frames(video_path: str, frame1: int, hist_func: Callable[[np.ndarray], np.ndarray] = grayscale_histogram):
    """
    Plots the four scenes and their histograms in the same window
    :param video_path: path to video file
    :param scene1: last frame index of the first scene
    :return: None
    """
    cap = cv2.VideoCapture(video_path)
    x = list(range(256))
    for i in range(4):
        frame_i = frame1 + i
        # get scene1 frame
        cap.set(1, frame_i)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("frame1 not found")
        hist = hist_func(bgr_to_grayscale(frame))
        plt.subplot(3, 4, 5 + i)
        plt.bar(x, hist)
        plt.title(f"frame {frame_i} histogram")
        plt.subplot(3, 4, 9 + i)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"frame {frame_i}")
        plt.axis("off")
    cap.release()


if __name__ == "__main__":
    for vp in [
        "video1_category1.mp4",
        "video2_category1.mp4",
        "video3_category2.mp4",
        "video4_category2.mp4",
        # "we_will_win_together.mp4"
    ]:
        scene1, scene2 = main(vp, 1)
        plt.title(f"Video: {vp}, Scene1: {scene1}, Scene2: {scene2}")
        plot_4_frames(vp, scene1 - 1, lambda x: cumsum_histogram(grayscale_histogram(x)))
        # plt.tight_layout()
        plt.show()