


class ComsolParams:
    def __init__(self):
       

    def load_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                self.__dict__[key.strip()] = float(value.strip())