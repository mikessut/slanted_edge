import numpy as np
from line import Line


def simulate_slanted_edge(shape=(768, 1024), low=2000, high=20000, dtype=np.uint16,
                          angle_deg=5, radius=150):
    
    angle = (90 + angle_deg) * np.pi / 180
    edge_vector = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    
    R, C = np.mgrid[:768, :1024]
    line = Line(np.array(shape[::-1]) // 2, norm_vec=edge_vector)
    D = line.dist_from_meshgrid(R, C)

    img = low * np.ones((768, 1024), dtype=np.uint16)

    img[(np.sqrt((R - 768//2)**2 + (C - 1024//2)**2) < radius) & (D < 0)] = high
    
    img = img.astype(np.uint16)

    return img