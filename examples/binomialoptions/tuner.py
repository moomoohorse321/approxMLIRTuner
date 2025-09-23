#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from builtins import range
from pprint import pprint
from accuracy import get_gt, compute_similarity, parse_pr_data
from dump import dump_data_to_csv


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
    "--program-cfg-default", help="override default program config exemplar location"
)
parser.add_argument(
    "--program-cfg-output", help="location final autotuned configuration is written"
)
parser.add_argument(
    "--program-settings", help="override default program settings file location"
)
parser.add_argument("--program-input", help="use only a given input for autotuning")
parser.add_argument(
    "--upper-limit", type=float, default=30, help="time limit to apply to initial test"
)
parser.add_argument("--test-config", action="store_true")
parser.add_argument(
    "--mlir-file", help="specify the MLIR file to parse for approxMLIR annotations"
)


class PetaBricksInterface(MeasurementInterface):
    def __init__(self, args):
        """
        Scope: internal to auto-tuner
        """
        self.program_settings = json.load(open(args.program_settings))
        objective = ThresholdAccuracyMinimizeTime(self.program_settings["accuracy"])
        input_manager = FixedInputManager()
        # pass many settings to parent constructor
        super(PetaBricksInterface, self).__init__(
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
                func_name = match.group(1)  # Everything before _type_
                param_type = match.group(2)  # 'threshold' or 'decision'
                param_index = int(match.group(3))  # The index number
                # Modify the MLIR file based on the parameter type and index
                if hasattr(self.args, "mlir_file") and self.args.mlir_file:
                    modification_success = self.modify_mlir_file(
                        key, value, func_name, param_type, param_index
                    )
                    # If modification failed due to validation, set a flag
                    if modification_success is False:
                        self.thresholds_validation_failed = True

    def modify_mlir_file(self, key, value, func_name, param_type, param_index):
        """
        Modify the MLIR file based on the configuration parameter
        """
        try:
            # Read the current MLIR file
            with open(self.args.mlir_file, "r") as f:
                mlir_content = f.read()

            # Find the specific function annotation
            pattern = rf'"approxMLIR\.util\.annotation\.decision_tree"\(\) <\{{[^}}]*func_name = "{func_name}"[^}}]*\}}'
            match = re.search(pattern, mlir_content, re.DOTALL)

            if match:
                annotation_content = match.group(0)
                modified = False

                if param_type == "threshold":
                    # Modify thresholds array
                    # Find the thresholds array for this function
                    thresholds_pattern = rf"thresholds = array<i32: ([^>]+)>"
                    thresholds_match = re.search(thresholds_pattern, annotation_content)

                    if thresholds_match:
                        thresholds_str = thresholds_match.group(1)
                        thresholds = [int(x.strip()) for x in thresholds_str.split(",")]

                        # Update the specific threshold at the given index
                        if param_index < len(thresholds):
                            thresholds[param_index] = value

                            # Check if thresholds array is in ascending order
                            if not self.is_thresholds_ascending(thresholds):
                                print(
                                    f"ERROR: Thresholds array is not in ascending order: {thresholds}"
                                )
                                return (
                                    False  # Return False to indicate validation failure
                                )

                            new_thresholds_str = ", ".join(map(str, thresholds))

                            # Replace in the annotation content using regex to be more specific
                            # This ensures we only replace the thresholds array in this specific annotation
                            thresholds_regex = (
                                rf"thresholds = array<i32: {re.escape(thresholds_str)}>"
                            )
                            new_thresholds_line = (
                                f"thresholds = array<i32: {new_thresholds_str}>"
                            )
                            annotation_content = re.sub(
                                thresholds_regex,
                                new_thresholds_line,
                                annotation_content,
                            )
                            modified = True

                elif param_type == "decision":
                    # Modify decisions array
                    decisions_pattern = rf"decisions = array<i32: ([^>]+)>"
                    decisions_match = re.search(decisions_pattern, annotation_content)

                    if decisions_match:
                        decisions_str = decisions_match.group(1)
                        decisions = [int(x.strip()) for x in decisions_str.split(",")]

                        # Update the specific decision at the given index
                        if param_index < len(decisions):
                            decisions[param_index] = value
                            new_decisions_str = ", ".join(map(str, decisions))

                            # Replace in the annotation content using regex to be more specific
                            # This ensures we only replace the decisions array in this specific annotation
                            decisions_regex = (
                                rf"decisions = array<i32: {re.escape(decisions_str)}>"
                            )
                            new_decisions_line = (
                                f"decisions = array<i32: {new_decisions_str}>"
                            )
                            annotation_content = re.sub(
                                decisions_regex, new_decisions_line, annotation_content
                            )
                            modified = True

                if modified:
                    # Replace the entire annotation in the MLIR content
                    mlir_content = mlir_content.replace(
                        match.group(0), annotation_content
                    )

                    # Write the modified content back to the file
                    with open(self.args.mlir_file, "w") as f:
                        f.write(mlir_content)

                    print(f"Successfully modified MLIR file: {key} = {value}")
                else:
                    print(f"Could not modify parameter {key} in MLIR file")
            else:
                print(f"Could not find function {func_name} in MLIR file")

        except Exception as e:
            print(f"Error modifying MLIR file: {e}")

    def is_thresholds_ascending(self, thresholds):
        """
        Check if the thresholds array is in ascending order
        """
        if len(thresholds) <= 1:
            return True

        for i in range(1, len(thresholds)):
            if thresholds[i] <= thresholds[i - 1]:
                return False
        return True

    def run_exec(self):
        run_cmd = []
        run_result = self.call_program(run_cmd)
        assert run_result["returncode"] == 0, run_result.get("stderr", "")
        return run_result

    def run(self, desired_result, input, limit):
        """
        Scope: environment dependent
        """
        limit = min(limit, self.args.upper_limit)
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

        gcc_cmd = [
            f"{os.environ.get('ROOT', '.')}/build/bin/polygeist-opt",
            f"{os.environ.get('ROOT', '.')}/tools/cgeist/Test/approxMLIR/approx_{os.environ.get('BENCH', 'binomialoptions')}.mlir",
            "-pre-emit-transform",
            "-emit-approx",
            "-config-approx",
            "-transform-approx",
            "-o",
            "test.mlir",
        ]

        compile_result = self.call_program(gcc_cmd)
        assert compile_result["returncode"] == 0

        cmd = [
            f"{os.environ.get('ROOT', '.')}/build/bin/cgeist",
            f"-resource-dir={os.environ.get('ROOT', '.')}/llvm-project/build/lib/clang/18",
            "-I",
            f"{os.environ.get('ROOT', '.')}/tools/cgeist/Test/polybench/utilities",
            "test.mlir",
            "-import-mlir",
            "-o",
            "test.exec",
        ]

        run_result = self.call_program(cmd)
        print("run_result", run_result)
        result = opentuner.resultsdb.models.Result()

        # run .exec
        run_result = self.run_exec()

        output = run_result["stdout"].decode("utf-8")
        time, out = parse_pr_data(output)

        result.time = time
        result.accuracy = compute_similarity(get_gt(), out)
        if result.time < limit + 3600:
            result.state = "OK"
        else:
            # time will be 2**31 if timeout
            result.state = "TIMEOUT"
        dump_data_to_csv(
            result.accuracy,
            result.time,
            desired_result.configuration.data,
            "binomialoptions.csv",
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

        # Parse MLIR file for approxMLIR annotations if provided
        if hasattr(self.args, "mlir_file") and self.args.mlir_file:
            self.choice_sites = self.parse_mlir_annotations(self.args.mlir_file)

        for name, choices in list(self.choice_sites.items()):
            print("name: ", name, "choices: ", choices)
            manipulator.add_parameter(LogIntegerParameter(name, choices[0], choices[1]))
        return manipulator

    def parse_mlir_annotations(self, mlir_file_path):
        """
        Parse MLIR file for approxMLIR.util.annotation.decision_tree annotations
        and convert them to a dictionary format
        """
        choice_sites = {}

        try:
            with open(mlir_file_path, "r") as f:
                mlir_content = f.read()

            # Find all approxMLIR.util.annotation.decision_tree annotations
            pattern = r'"approxMLIR\.util\.annotation\.decision_tree"\(\) <\{([^}]+)\}'
            matches = re.findall(pattern, mlir_content, re.DOTALL)

            for match in matches:
                # Extract function name
                func_name_match = re.search(r'func_name = "([^"]+)"', match)
                if not func_name_match:
                    continue
                func_name = func_name_match.group(1)

                # Extract num_thresholds
                num_thresholds_match = re.search(r"num_thresholds = (\d+)", match)
                if not num_thresholds_match:
                    continue
                num_thresholds = int(num_thresholds_match.group(1))

                # Extract thresholds_lowers array
                thresholds_lowers_match = re.search(
                    r"thresholds_lowers = array<i32: ([^>]+)>", match
                )
                if thresholds_lowers_match:
                    thresholds_lowers_str = thresholds_lowers_match.group(1)
                    thresholds_lowers = [
                        int(x.strip()) for x in thresholds_lowers_str.split(",")
                    ]
                else:
                    continue

                # Extract thresholds_uppers array
                thresholds_uppers_match = re.search(
                    r"thresholds_uppers = array<i32: ([^>]+)>", match
                )
                if thresholds_uppers_match:
                    thresholds_uppers_str = thresholds_uppers_match.group(1)
                    thresholds_uppers = [
                        int(x.strip()) for x in thresholds_uppers_str.split(",")
                    ]
                else:
                    continue

                # Extract decision_values array
                decision_values_match = re.search(
                    r"decision_values = array<i32: ([^>]+)>", match
                )
                if decision_values_match:
                    decision_values_str = decision_values_match.group(1)
                    decision_values = [
                        int(x.strip()) for x in decision_values_str.split(",")
                    ]
                else:
                    continue

                # Create dictionary entries for thresholds
                # Use the actual length of the arrays instead of num_thresholds
                for i in range(len(thresholds_lowers)):
                    if i < len(thresholds_uppers):
                        key = f"{func_name}_threshold_{i}"
                        value = [thresholds_lowers[i], thresholds_uppers[i]]
                        choice_sites[key] = value

                # Create dictionary entries for decisions
                # Extract decisions array if it exists
                decisions_match = re.search(r"decisions = array<i32: ([^>]+)>", match)
                if decisions_match:
                    decisions_str = decisions_match.group(1)
                    decisions = [int(x.strip()) for x in decisions_str.split(",")]

                    # Create decision entries with ranges based on decision_values
                    for i, decision in enumerate(decisions):
                        key = f"{func_name}_decision_{i}"
                        # Use the full range of decision_values for each decision
                        value = [min(decision_values), max(decision_values)]
                        choice_sites[key] = value
                else:
                    # Fallback to original behavior if no decisions array
                    for i, decision_value in enumerate(decision_values):
                        key = f"{func_name}_decision_{i}"
                        value = [decision_value]
                        choice_sites[key] = value

        except FileNotFoundError:
            log.warning(f"MLIR file not found: {mlir_file_path}")
        except Exception as e:
            log.error(f"Error parsing MLIR file: {e}")

        return choice_sites

    def test_config(self):
        pprint(self.manipulator.random())


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.program_cfg_default:
        args.program_cfg_default = args.program + ".cfg.default"
    if not args.program_cfg_output:
        args.program_cfg_output = args.program + ".cfg"
    if not args.program_settings:
        args.program_settings = args.program + ".settings"
    if args.test_config:
        PetaBricksInterface(args).test_config()
    else:
        PetaBricksInterface.main(args)
