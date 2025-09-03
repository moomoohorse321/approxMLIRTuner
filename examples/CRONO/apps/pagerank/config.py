def config(config_file="pagerank.settings"):
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
    os.environ["BENCH"] = "blackscholes"
    os.environ["ROOT"] = "/Users/yi/Desktop/PolygeistSample"

    # Add the project root to Python path
    sys.path.insert(0, "/Users/yi/Desktop/approxMLIRTuner")
    accuracy = np.linspace(0.6, 1.0, 4)
    for acc in accuracy:
        print("Accuracy: ", acc)
        with open(config_file, "w") as f:
            import json

            json.dump({"accuracy": acc}, f)
        # run the pagerank program
        os.system(
            f"PYTHONPATH=/Users/yi/Desktop/approxMLIRTuner python tuner.py pagerank --mlir-file /Users/yi/Desktop/PolygeistSample/tools/cgeist/Test/approxMLIR/approx_blackscholes.mlir --database opentuner.db --stop-after=1200"
        )


if __name__ == "__main__":
    config()
