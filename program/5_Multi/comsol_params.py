
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent
params_dir = BASE_DIR/'comsol_parameters'

class ComsolParams:
    def __init__(self, global_file=params_dir/'global_params.txt', component_file=params_dir/'component_variables.txt'):
        self.params = {}
        self.load_file(global_file)
        self.load_file(component_file)

    def load_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue  # Skip empty lines and comments
                
                parts = self.split_line(line)
                key = parts[0]
                value = parts[1]
                self.params[key] = value

                setattr(self, key, value)   # To use as attribute: comsol_params.epss_neg

    def split_line(self, line):
        """
        Handles:
        epss_neg 0.6 "description"
        """
        # Split but keep quoted strings together
        return re.findall(r'"[^"]*"|\S+', line)
    
    # def get_param(self, key):
    #     return self.params.get(key)
    
if __name__ == "__main__":
    comsol_params = ComsolParams()
    print(comsol_params.L_sep)  # Access as attribute