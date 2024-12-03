import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import least_squares
from line import Line


def get_edge_pixels(img):
    """
    This assumes a nominally vertical edge
    """
 
    edge = img.astype(float)
    kernel_size = 5
    kernel = np.ones((1, kernel_size))
    kernel[0, :(kernel_size - 1) // 2] = -1
    kernel[0, kernel_size // 2] = 0

    edge = np.abs(convolve(edge, kernel))
    edge = edge > 0.5 * edge
    return edge


def fit_edge_to_line(edge):
    """
    Also assumes a nominally vertical edge
    """
    def err(X, pt0y, pt1y, rows, cols):
        pt0x = X[0]
        pt1x = X[1]
        return Line([pt0x, pt0y], [pt1x, pt1y]).dist(np.column_stack([cols, rows])) ** 2

    rows, cols = np.where(edge)
    pt0y = 100
    pt1y = 0
    X = least_squares(err, [0, 0], args=(pt0y, pt1y, rows, cols))
    pt0 = [X.x[0], pt0y]
    pt1 = [X.x[1], pt1y]
    return Line(pt0, pt1)


def line_spread_function(img: np.ndarray, line: Line, N_over_sample = 8):
    R, C = np.mgrid[:img.shape[0], :img.shape[1]]

    D  = line.dist_from_meshgrid(R, C)

    extent = (np.ceil(D.min()), np.floor(D.max()))
    
    x = np.linspace(*extent, int(np.diff(extent)[0] * N_over_sample))
    y = np.zeros(x.shape)

    dig = np.digitize(D.flatten(), x[:-1])

    for u in np.unique(dig):
        y[u] = img.flatten()[dig == u].mean()

    extent = (-np.min(np.abs(extent)), np.min(np.abs(extent)))
    idx = (x > extent[0]) & (x < extent[1])
    x = x[idx]
    y = y[idx]

    dy = np.diff(y)
    dy = dy * np.hamming(len(dy))

    return x, y, dy


def mtf(dy, N_over_sample=8):
    Y = np.fft.fft(dy)
    Y = Y[:len(Y) // 2]

    fs = N_over_sample  # [=] samples / pix

    f = np.arange(len(Y)) / len(Y) * N_over_sample
    Y /= np.abs(Y[0])
    return f, Y