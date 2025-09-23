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
    os.environ["BENCH"] = "pagerank"
    os.environ["ROOT"] = "/home/yimu3/PolygeistSample"

    # Add the project root to Python path
    sys.path.insert(0, "/home/yimu3/approxMLIRTuner")
    accuracy = np.linspace(0.6, 1.0, 4)
    for acc in accuracy:
        print("Accuracy: ", acc)
        with open(config_file, "w") as f:
            import json

            json.dump({"accuracy": acc}, f)
        # run the pagerank program
        os.system(
            f"PYTHONPATH=/home/yimu3/approxMLIRTuner python /home/yimu3/approxMLIRTuner/examples/pagerank/tuner.py pagerank --mlir-file /home/yimu3/approxMLIRTuner/examples/pagerank/approx_pagerank.mlir --database opentuner.db --stop-after=1200"
        )


if __name__ == "__main__":
    config()
