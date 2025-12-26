import subprocess

def run_bash(bash_name):
    subprocess.run(bash_name, capture_output=False)
    return 0