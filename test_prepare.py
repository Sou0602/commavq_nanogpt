import os

# Create dummy binary files from 1.bin to 40.bin
dummy_directory = os.path.dirname(__file__) + "dummy_binaries"

if not os.path.exists(dummy_directory):
    os.makedirs(dummy_directory)

for i in range(0, 41):
    with open(f"{dummy_directory}/{i}.bin", "wb") as file:
        file.write(bytes([i] * 1024))

# Execute the commands
concatenation_command = f'for i in $(seq 0 39); do cat "{dummy_directory}/$i.bin" >> "{dummy_directory}/train.bin"; rm "{dummy_directory}/$i.bin"; done'
os.system(concatenation_command)

rename_command = f'cp "{dummy_directory}/40.bin" "{dummy_directory}/val.bin"'
os.system(rename_command)

remove_command = f'rm "{dummy_directory}/40.bin"'
os.system(remove_command)

files = os.listdir(dummy_directory)

if len(files) == 2:
    files.sort()
    assert files[0] == "train.bin" and files[1] == "val.bin"
    print(f"Test Passed!")

else:
    print(f"Test Failed!")
    print(files)
os.system(f"rm -r {dummy_directory}")
