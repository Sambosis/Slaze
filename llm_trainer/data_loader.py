import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.filenames = []

    def load_source_files(self):
        # In a real implementation, you would read the content of the .py files
        # For now, we'll just collect the filenames
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.py'):
                self.filenames.append(filename)

    def preprocess_data(self):
        # In a real implementation, you would tokenize and prepare the data for the model
        pass

    def get_filenames(self):
        return self.filenames