import math

import matplotlib.pylab as plt


def plot_album_covers(
    album_cover_list: list, kind: str = "row", facecolor: str = "black"
) -> plt.Figure:
    """
    Plot a collection of selected album covers, either as a square (kind == 'square') or row (kind == 'row').

    Parameters
    ----------
    album_cover_list : list[np.array]
        List of selected album covers.
    kind : str
        The kind of plot, either square or row.
    facecolor : str
        The background colour for the collage of album covers.

    Returns
    -------
    fig : plt.Figure :
        The figure object.
    """
    if kind == "row":
        n = len(album_cover_list)
        fig, axes = plt.subplots(1, n, figsize=(15, 15), facecolor=facecolor)
    elif kind == "square":
        assert math.sqrt(
            len(album_cover_list)
        ).is_integer(), "Album collage is a square"
        n = int(math.sqrt(len(album_cover_list)))
        fig, axes = plt.subplots(n, n, figsize=(15, 15), facecolor=facecolor)
    else:
        raise Exception(f"kind = '{kind}' is not supported.")
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(album_cover_list[i])
        ax.axis("off")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig
