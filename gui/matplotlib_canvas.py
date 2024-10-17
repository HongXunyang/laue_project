from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setObjectName("matplotlib_canvas")
        self.axes = self.fig.add_subplot(111)

    def plot(self, data):
        self.axes.clear()
        self.axes.plot(data)
        self.draw()
