#!/usr/bin/env python
import argparse
import json
import logging
import os
import re
import subprocess
from pprint import pprint
from dump import dump_data_to_csv
from paths import TUNER, ROOT, BENCH
from approxMLIR import ApproxMLIRSDK
SDK: ApproxMLIRSDK = ApproxMLIRSDK("./binary", "./MLIR", ROOT)

import opentuner
from opentuner import (
    ConfigurationManipulator,
    IntegerParameter,
    LogIntegerParameter,
    SwitchParameter,
)

from opentuner import MeasurementInterface
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.search.objective import ThresholdAccuracyMinimizeTime

log = logging.getLogger("pbtuner")

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument("program", help="PetaBricks binary program to autotune")
parser.add_argument(
    "--program-settings", help="override default program settings file location"
)


class ApproxTunerInterface(MeasurementInterface):
    def __init__(self, args):
        """
        Scope: internal to auto-tuner
        """
        self.program_settings = json.load(open(args.program_settings))
        SDK.populate_orchestration_knobs(2)
        objective = ThresholdAccuracyMinimizeTime(self.program_settings["accuracy"])
        input_manager = FixedInputManager()
        # pass many settings to parent constructor
        super(ApproxTunerInterface, self).__init__(
            args,
            program_name=args.program,
            objective=objective,
            input_manager=input_manager,
        )

    def build_config(self, cfg):
        """
        Scope: internal to auto-tuner
        """
        r = []

        # Parse MLIR file for decision tree annotations
        for key, value in cfg.items():
            print(f"Processing config: {key} = {value}")

            # Parse the key to understand what needs to be modified
            # Format: func_name_type_index (e.g., compute_cndf_threshold_0, compute_cndf_decision_1)
            # Use regex to find the pattern: (.*)_(threshold|decision)_(\d+)
            pattern = r"^(.+)_(threshold|decision)_(\d+)$"
            match = re.match(pattern, key)

            if match:
                func_name = match.group(1).split(".")[1][5:]  # Everything before _type_
                param_type = match.group(2)  # 'threshold' or 'decision'
                param_index = int(match.group(3))  # The index number
                # Modify the MLIR file based on the parameter type and index
                file_path = SDK.get_mlir_files(key.split(".")[0] + ".mlir")
                modification_success = SDK.modify_mlir_file(
                    key, value, func_name, param_type, param_index, file_path
                )
                # If modification failed due to validation, set a flag
                if modification_success is False:
                    raise ValueError("Failed due to validation, SDK.modify_mlir_file")

    def run(self, desired_result, input, limit):
        """
        Scope: environment dependent
        """
        # Reset validation flag before building config
        self.thresholds_validation_failed = False

        self.build_config(desired_result.configuration.data)
        # Check if thresholds validation failed during MLIR modification
        if self.thresholds_validation_failed:
            result = opentuner.resultsdb.models.Result()
            result.state = "ERROR"
            result.time = float("inf")  # Set a very high time to indicate failure
            result.accuracy = 0.0  # Set accuracy to 0 for invalid configuration
            print(
                "Configuration rejected due to invalid thresholds (not in ascending order)"
            )
            return result

        # Process each MLIR file
        mlir_files = SDK.get_mlir_files()
        for mlir_file in mlir_files:
            exec_path = SDK.compile_mlir(mlir_file)

        result = opentuner.resultsdb.models.Result()

        total_time = 0
        total_accuracy = 0
        exec_count = 0

        run_cmd = [
            "python3",
            "jax_tool.py",
        ]
        run_result = self.call_program(run_cmd)
        print(run_result)
        assert run_result["returncode"] == 0

        # Parse output from direct_rag_agent.py
        output = run_result["stdout"].decode("utf-8")

        # Look for TUNER_OUTPUT line
        time = 0.0
        accuracy = 0.0

        for line in output.split("\n"):
            if line.startswith("TUNER_OUTPUT:"):
                # Parse: TUNER_OUTPUT: time=1.234567 accuracy=0.876543
                parts = line.split()
                for part in parts:
                    if part.startswith("time="):
                        time = float(part.split("=")[1])
                    elif part.startswith("accuracy="):
                        accuracy = float(part.split("=")[1])
                break

        result.time = time
        result.accuracy = accuracy

        if result.time < limit + 3600:
            result.state = "OK"
        else:
            # time will be 2**31 if timeout
            result.state = "TIMEOUT"
        dump_data_to_csv(
            result.accuracy,
            result.time,
            desired_result.configuration.data,
           f"{BENCH}.csv",
        )

        return result

    def save_final_config(self, configuration):
        """
        called at the end of autotuning with the best
        resultsdb.models.Configuration

        Scope: internal to auto-tuner
        """
        self.manipulator().save_to_file(configuration.data, "approx_config.json")

    def manipulator(self):
        """
        create the configuration manipulator, from example config

        Scope: internal to auto-tuner
        """
        manipulator = ConfigurationManipulator()
        # print(manipulator().random())
        self.choice_sites = dict()

        # Parse MLIR files for approxMLIR annotations if provided
        mlir_files = SDK.get_mlir_files()
        for mlir_file in mlir_files:
            file_choice_sites = SDK.parse_mlir_annotations(mlir_file)
            self.choice_sites.update(file_choice_sites)

        for name, choices in list(self.choice_sites.items()):
            print("name: ", name, "choices: ", choices)
            manipulator.add_parameter(IntegerParameter(name, choices[0], choices[1]))
        return manipulator

    def test_config(self):
        pprint(self.manipulator.random())


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.program_settings:
        args.program_settings = args.program + ".settings"
    ApproxTunerInterface.main(args)
