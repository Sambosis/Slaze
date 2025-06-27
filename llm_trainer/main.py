import sys
from PyQt5.QtWidgets import QApplication
from gui import TrainingGUI
from model import TransformerModel
from data_loader import DataLoader

def main():
    app = QApplication(sys.argv)
    
    # Configuration for the model and data
    config = {
        'num_layers': 4,
        'd_model': 128,
        'num_heads': 8,
        'dff': 512,
        'input_vocab_size': 10000,  # Example size
        'target_vocab_size': 10000, # Example size
        'pe_input': 1000,
        'pe_target': 1000,
        'rate': 0.1
    }
    
    # Initialize components
    data_loader = DataLoader(data_dir='.venv') # a an example directory
    transformer = TransformerModel(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        dff=config['dff'],
        input_vocab_size=config['input_vocab_size'],
        target_vocab_size=config['target_vocab_size'],
        pe_input=config['pe_input'],
        pe_target=config['pe_target'],
        rate=config['rate']
    )
    
    # Create and show the GUI
    main_window = TrainingGUI(data_loader, transformer)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()