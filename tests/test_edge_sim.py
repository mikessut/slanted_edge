import matplotlib.pyplot as plt
from slanted_edge_sim import simulate_slanted_edge
import os
import pytest


PLOTS = os.environ.get('PYTEST_PLOTS', False)


@pytest.fixture(scope='package')
def plot_figs():
    yield None
    # This runs as finalization of fixture
    if PLOTS:
        plt.show()


def test_sim_edge(plot_figs):

    img = simulate_slanted_edge()

    plt.imshow(img, vmin=0, vmax=0xffff)