from ex1 import *
import matplotlib.pyplot as plt


def __plot(x):
    import matplotlib.pyplot as plt
    x = list(x)
    plt.plot(x)
    return x


main.types[3] = [
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

def __plot_2_scences(video_path: str, scene1: int, scene2: int):
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
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.title("Scene 1")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    plt.title("Scene 2")
    plt.show()


if __name__ == "__main__":
    for vp in [
        "video1_category1.mp4",
        "video2_category1.mp4",
        "video3_category2.mp4",
        "video4_category2.mp4"
    ]:
        scene1, scene2 = main(vp, 2)
        print(f"video: {vp}, scene1: {scene1}, scene2: {scene2}")
        __plot_2_scences(vp, scene1, scene2)