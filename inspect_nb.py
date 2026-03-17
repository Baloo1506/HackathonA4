import json, sys

nb_path = r'C:\Users\gaspa\Desktop\Git\HackathonA4\hr_attribution_7.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])[:120].replace('\n', ' ')
        print(f'Cell {i}: {src}')
