import subprocess

for volume in range(1, 2):
    for slice in range(1, 2):
        print(f"Running algorithm_statistics.py with volume={volume} and slice={slice}")
        subprocess.run([
            "python", "algorithm_statistics.py",
            "--volume", str(volume),
            "--slice", str(slice)
        ])