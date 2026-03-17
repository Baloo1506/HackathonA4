import json

nb_path = r'C:\Users\gaspa\Desktop\Git\HackathonA4\hr_attribution_7.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print full source of cells 24-35 (model training/selection area)
for i in [24, 25, 26, 27, 28, 31, 32, 33, 34, 35]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        print(f'\n====== Cell {i} ({c["cell_type"]}) ======')
        print(''.join(c['source']))
