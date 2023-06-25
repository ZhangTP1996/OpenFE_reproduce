import os

print(os.listdir('shell_inst'))

for sh in os.listdir('shell_inst'):
    os.system(f'bash shell_inst/{sh}')