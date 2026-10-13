import subprocess

for volume in range(1, 10):
    for slice in range(1, 4):
        print(f"Running main.py with volume={volume} and slice={slice}")
        subprocess.run([
            "python", "main.py",
            "--volume", str(volume),
            "--slice", str(slice)
        ])