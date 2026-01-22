import numpy as np

NUM_TRIALS = 20

# with open('trial1_data', 'w') as f:
min_depths = []
max_depths = []
for i in range(1, NUM_TRIALS + 1):
    file_path = './preliminary_data/trial' + str(i) + '.npy'
    data = np.load(file_path, allow_pickle=True)
    payload = data.item()

    print(f'\n-- Data Entries: --')
    for key, value in payload.items():
        dim = None
        if isinstance(value, np.ndarray):
            dim = value.shape
        else:
            dim = len(value)

        print(f'    {key} with dimensions {dim}')

    depth_image = payload['depth']
    depth_array = depth_image.flatten()
    rgb_image = payload['rgb']

    min_depth = min(depth_array)
    max_depth = max(depth_array)

    min_depths.append(min_depth)
    max_depths.append(max_depth)

    print(f'\n    {file_path} Depth Values')
    print(f'    Depth min: {str(min_depth)}, max: {str(max_depth)}')

print(f'\nMax depth: {max(max_depths)}')
print(f'\nMin depth: {max(min_depths)}')