from PyQt5.QtCore import QObject, pyqtSignal


class ClosePackingWorker(QObject):
    progress_signal = pyqtSignal(float, float, float)
    finished_signal = pyqtSignal(
        object, list
    )  # Pass sampleholder and area_evolution_list

    def __init__(self, sampleholder, kwargs):
        super().__init__()
        self.sampleholder = sampleholder
        self.kwargs = kwargs

    def run(self):
        from close_packing import (
            batch_optimization,
        )  # Import here to avoid circular imports

        # Run batch_optimization with progress reporting
        _, _, _, area_evolution_list = batch_optimization(
            self.sampleholder,
            progress_callback=self.progress_signal.emit,
            **self.kwargs,
        )
        # Emit the finished signal with results
        self.finished_signal.emit(self.sampleholder, area_evolution_list)

    def progress_callback(self, progress, estimated_total_time, remaining_time):
        # Emit the progress_signal with timing information
        self.progress_signal.emit(progress, estimated_total_time, remaining_time)
