"""
    Module with all functions that generate visualizations.
"""

from matplotlib import animation
import matplotlib.pyplot as plt
import os


def save_frames_as_gif(frames: list, path: str = './',
                       filename: str = 'gym_animation.gif'):
    """
    Save frames as gif.

    Ensure you have imagemagick installed .

    Args:
        frames (list): List with all rgb_arrays of the frames.
        path (str, optional): Path to save the gif. Defaults to './'.
        filename (str, optional): _Name of the gif file.
                            Defaults to 'gym_animation.gif'.
    """

    plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72))

    patch = plt.imshow(frames[0])
    plt.axis('off')

    anim = animation.FuncAnimation(
        plt.gcf(),
        lambda i: patch.set_data(frames[i]),
        frames=len(frames),
        interval=50)

    anim.save(os.path.join(path, filename), writer='imagemagick', fps=60)
