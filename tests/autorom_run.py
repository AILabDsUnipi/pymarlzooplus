import subprocess


def run_autorom_with_auto_yes():
    command = ["AutoROM", "--accept-license"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Send "Y" followed by a newline to the stdin of the process
        process.stdin.write("Y\n")
        process.stdin.flush()
    except ValueError as e:
        print(f"Error writing to stdin: {e}")

    stdout, stderr = process.communicate(input=None, timeout=30)

    print("Stdout:")
    print(stdout)
    print("Stderr:")
    print(stderr)

    return process.returncode


if __name__ == "__main__":
    return_code = run_autorom_with_auto_yes()
    print(f"AutoROM exited with return code: {return_code}")
