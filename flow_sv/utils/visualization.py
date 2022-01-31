"""
Loss functions for optical flow learning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

def flow_to_rgb(flow):
    '''
    Function from this answer
    https://stackoverflow.com/a/49636438

    input: optical flow [H, W, 2]
    output: RGB image [H, W, 3]
    '''

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=np.uint8)
    hsv[..., 2] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = (np.clip(mag / 20, 0, 1) * 255).astype('uint8')
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def plot_poses(poses, ax=None, view='xz', **kwargs):
    'Plot the 2D view of the trajectory'
    # Transform from camera coords to world coords
    coords = poses @ np.array([0,0,0,1])
    m_x, m_y, m_z = coords.T[:3]

    # Plot
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect('equal')
    view_dict = dict(x=m_x, y=m_y, z=m_z)
    ax.plot(view_dict[view[0]], view_dict[view[1]], **kwargs)

    return ax

def confidence_ellipse(mean_x, mean_y, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    mean_x, mean_y : float
        Position of the ellipse

    cov : float
        Covariance

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
