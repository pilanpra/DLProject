import subprocess

def install_required_libraries():
    subprocess.check_call(["pip", "install", "transformers"])
    subprocess.check_call(["pip", "install", "torch"])
    subprocess.check_call(["pip", "install", "datasets"])
    subprocess.check_call(["pip", "install", "matplotlib"])


