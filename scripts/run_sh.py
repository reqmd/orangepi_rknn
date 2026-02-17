import subprocess

def run_command(cmd):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            check=True,
            text=True
        )
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"

def run_check_call(args):
    try:
        result = subprocess.check_output(args=args, text=True)
        return result
    except Exception as e:
        return f"ERROR: {e}"