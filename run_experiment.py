# run your specific model here. edit within experiments/name/file.py

import argparse
import os
from subprocess import call


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str) #etc
    parser.add_argument("file", type=str)
    parser.add_argument("--save", type=bool)
    args = parser.parse_args()

    fpath = os.path.join(os.getcwd(), "ue", "experiments", args.name, args.file+".py")
    # check if file exists
    if os.path.isfile(fpath):
        try:
            call(["python", fpath])
            # # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            # spec = importlib.util.spec_from_file_location("experiments", fpath)
            # foo = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(foo)
            # foo.run()
        except:
            print("execution failed")
    else:
        print("path given not found...error")
    

print("yeet")