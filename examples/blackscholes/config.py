from paths import TUNER, ROOT, BENCH
def config(config_file=f"{BENCH}.settings"):
    """
    An example of *.settings
    {
        "accuracy": 0.9
    }


    """
    import numpy as np
    import os
    import sys

    # Set global environment variables
    os.environ["BENCH"] = BENCH
    os.environ["ROOT"] = ROOT

    # Add the project root to Python path
    sys.path.insert(0, TUNER)
    accuracy = np.linspace(0.6, 1.0, 4)
    for acc in accuracy:
        print("Accuracy: ", acc)
        with open(config_file, "w") as f:
            import json

            json.dump({"accuracy": acc}, f)
        os.system(
            f"PYTHONPATH={TUNER} python tuner.py {BENCH} --mlir-file {TUNER}/examples/{BENCH}/approx_{BENCH}.mlir --database opentuner.db --stop-after=1200"
        )


if __name__ == "__main__":
    config()
