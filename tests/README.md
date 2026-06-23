# Test Generation Suite

Generates ONNX models together with their input/output data for a set of operators. The engine loads the resulting `tests.json` to run each model and validate its output against the expected values.

## Setup

```bash
cd tests
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

By default the artifacts are written to `tests/generated_tests/`. Pass `--base_path` to change the output location:

```bash
python main.py --base_path ../demo/addons/ml/tests
```

## Output

`generated_tests/` contains one `.onnx` file per test case and a single `tests.json` that lists every case with its name, input data, input shape, and expected output.