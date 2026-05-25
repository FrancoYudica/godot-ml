import json

class TestData:
    def __init__(
            self, 
            name: str, 
            input_data: list, 
            input_shape: list, 
            output_data: list):
        
        self.name = name    
        self.input_data = input_data
        self.input_shape = input_shape
        self.output_data = output_data

    def to_dict(self):
        """Converts the object to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "input_data": self.input_data,
            "input_shape": self.input_shape,
            "expected_output": self.output_data
        }

def build_filename(name, **kwargs):
    separator = "_"
    param_enter = "("
    param_close = ")"
    param_assign = "="

    filename = name

    for i in range(len(kwargs)):
        key, value = list(kwargs.items())[i]
        filename += f"{separator}{param_enter}{key}{param_assign}{value}{param_close}"

    return filename


class TestBuilder:
    
    def __init__(self, base_path, name):
        self.name = name
        self.model_filepath = f"{base_path}/{self.name}.onnx"

    def create_model(self):
        raise NotImplemented()

    def create_test_data(self) -> TestData:
        raise NotImplemented()

class TestGenerator:
    def __init__(self):
        self.tests = []

    def add_test(self, test: TestData):
        self.tests.append(test)

    def export_to(self, filepath: str):
        """Exports the entire suite as a single JSON array."""
        data_to_export = [test.to_dict() for test in self.tests]
        with open(filepath, 'w') as f:
            json.dump(data_to_export, f, indent=4)
        print(f"[*] Successfully exported {len(self.tests)} tests to {filepath}")