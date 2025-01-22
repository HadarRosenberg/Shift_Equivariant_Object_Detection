import subprocess

# configs = ["-is_afc -fpn", "-is_afc", ""]
configs = ["-is_afc --only-fpn"]
# crop_type = ["-is_cyclic", ""]
crop_type = ["-is_cyclic"]

for max_shift in [5]:
    base_command = ["python", "test_wrapper.py", "--threads", "7", "--max_shift", str(max_shift)]
    for config in configs:
        for crop in crop_type:
            # Split the config and crop strings into arguments and add them
            command = base_command + config.split() + crop.split()
            try:
                subprocess.call(command)
                print(f"Finished running: {' '.join(command)}")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e}")
