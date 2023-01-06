def make_clickable_df(val):
    """
    Wrap value into an HTML tag in order to allow the value to be clickable if rendered as HTML.

    Parameters
    ----------
    val

    Returns
    -------
    """
    return '<a target="_blank" href="{}">{}</a>'.format(val, "preview")


def make_clickable_html(val, version=2):
    """
    Wrap value into an HTML tag that allows audio to be played.

    Parameters
    ----------
    val
    version

    Returns
    -------
    """
    v1 = """
    <audio controls="controls">
        <source src="{}" type="audio/mpeg"/>
    </audio>
    """
    v2 = """
    <iframe 
        src="{}" width="80" height="30" frameborder="0" allowtransparency="true" allow="encrypted-media"> 
    </iframe>
    """
    versions = {1: v1, 2: v2}
    return versions[version].format(val)
