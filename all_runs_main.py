import subprocess

for volume in range(1, 10):
    for slice in range(1, 4):
        subprocess.run([
            "python", "main.py",
            "--volume", str(volume),
            "--slice", str(slice)
        ])