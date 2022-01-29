import matplotlib.pylab as plt
import math


def plot_album_covers(album_cover_list: list, kind: str = "row"):
    """
    Plots a collection of selected album covers, either as a square (kind == 'square') or row (kind == 'row').

    Parameters
    ----------
    album_cover_list : list[np.array]
        List of selected album covers.
    kind : str
        The kind of plot, either square or row.

    Returns
    -------
    fig : matplotlib figure
    """
    if kind == "row":
        n = len(album_cover_list)
        fig, axes = plt.subplots(1, n, figsize=(15, 15), facecolor='black')
    elif kind == "square":
        assert math.sqrt(len(album_cover_list)).is_integer(), "Album collage is a square"
        n = int(math.sqrt(len(album_cover_list)))
        fig, axes = plt.subplots(n, n, figsize=(15, 15), facecolor="black")
    else:
        raise Exception(f"kind = '{kind}' is not supported.")
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(album_cover_list[i])
        ax.axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig
