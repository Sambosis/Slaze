from PyQt5.QtWidgets import QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal

# Worker thread for training
class TrainingThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, data_loader, model):
        super().__init__()
        self.data_loader = data_loader
        self.model = model

    def run(self):
        self.update_signal.emit('Starting data loading...')
        self.data_loader.load_source_files()
        self.update_signal.emit('Data loaded. Preprocessing...')
        self.data_loader.preprocess_data()
        self.update_signal.emit('Preprocessing complete. Starting training...')
        # Placeholder for training logic
        # In a real scenario, you would use the preprocessed data to train the model
        # and emit updates on the progress.
        for epoch in range(5):
            self.update_signal.emit(f'Epoch {epoch + 1}/5')
            # Simulate training time
            self.msleep(2000)
        self.update_signal.emit('Training finished!')

class TrainingGUI(QMainWindow):
    def __init__(self, data_loader, model):
        super().__init__()
        self.setWindowTitle('LLM Training Progress')
        self.setGeometry(100, 100, 600, 400)
        
        self.data_loader = data_loader
        self.model = model

        # UI Elements
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.start_button = QPushButton('Start Training')
        
        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_area)
        layout.addWidget(self.start_button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connections
        self.start_button.clicked.connect(self.start_training)

    def start_training(self):
        self.start_button.setEnabled(False)
        self.training_thread = TrainingThread(self.data_loader, self.model)
        self.training_thread.update_signal.connect(self.update_progress)
        self.training_thread.start()

    def update_progress(self, message):
        self.text_area.append(message)
