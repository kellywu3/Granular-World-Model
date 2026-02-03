import numpy as np
import os

# Takes the "first" and "last" depth images offset by a time buffer
# Concatenates the image to a list and places them into input_data and label_data respectively as a dict
def save_data():
    folder_path = 'preliminary_data'
    filenames = os.listdir(folder_path)

    input_folder = 'inputs'
    action_folder = 'actions'
    label_folder = 'labels'
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(action_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for idx in range(len(filenames)):
        file_path = os.path.join(folder_path, filenames[idx])
        data = np.load(file_path, allow_pickle=True)
        payload = data.item()

        depth_images = payload['depth']

        depth_input = depth_images[3]
        depth_label = depth_images[-4]
        action = np.array([
                    0.0,
                    -0.53,
                    0.7,
                    0.0,
                    -1.315,
                    0.7,
                    0.785,
                    -1.315,
                    0.7,
                    0.785,
                    -0.53,
                    0.7,
                    0.785,
                    0.255,
                    0.7,
                    0.0,
                    0.255,
                    0.7,
                    0.0,
                    -0.53,
                    0.7,
                ])

        input_path = os.path.join(input_folder, f'input_{idx}')
        action_path = os.path.join(action_folder, f'action_{idx}')
        label_path = os.path.join(label_folder, f'label_{idx}')

        np.save(input_path, depth_input)
        np.save(action_path, action)
        np.save(label_path, depth_label)

if __name__ == '__main__':
    save_data()