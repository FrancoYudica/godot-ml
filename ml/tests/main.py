from core import TestGenerator, TestData, TestBuilder
from operators.gemm_tests import *
from operators.relu_tests import *
from operators.sigmoid_tests import *
from operators.conv_2d_tests import *
from operators.conv_transpose_tests import *
from operators.max_pool_2d_tests import *
from operators.im2col_tests import *
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML engine test suite.")

    folder_name = "generated_tests"

    parser.add_argument(
        "--base_path", 
        type=str, 
        default="", 
        help="Directory to save generated .onnx models."
    )
    args = parser.parse_args()

    base_path = args.base_path + "/" + folder_name


    # 2. Use arguments
    test_generator = TestGenerator()

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    builders = [
        GemmIdentity(base_path),
        ReLuBasic(base_path),
        SigmoidBasic(base_path),
        Conv2DTest1(base_path),
        ConvTranspose2x2(base_path),
        ConvTransposeDilations2(base_path),
        ConvTranspose4x4(base_path),
        MaxPoolTest0(base_path),
        MaxPoolTest1(base_path),
        MaxPoolTest2(base_path),
        MaxPoolTest3(base_path),
        MaxPoolTest4(base_path),
        Im2ColTest1(base_path),
        Im2ColTest2(base_path),
        Im2ColTest3(base_path),
        Im2ColTest4(base_path),
    ]

    for builder in builders:
        print(f"[*] Building {builder.name}...")
        builder.create_model()
        
        if not os.path.exists(builder.model_filepath):
            print(f"Failed to create model: {builder.model_filepath}")
            continue
            
        test_generator.add_test(builder.create_test_data())

    # 3. Export to the dynamic path
    test_generator.export_to(base_path + "/tests.json")
    print(f"[+] Successfully exported test suite to: {base_path + '/tests.json'}")