from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setObjectName("matplotlib_canvas")
        self.ax_sampleholder = self.fig.add_subplot(121)
        self.ax_evolution = self.fig.add_subplot(122)
        # remove all ticks on both axes
        self.ax_sampleholder.set_xticks([])
        self.ax_sampleholder.set_yticks([])
        self.ax_evolution.set_xticks([])
        self.ax_evolution.set_yticks([])
