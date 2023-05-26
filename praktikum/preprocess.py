import numpy as np
import torch


def add_grok_tip(dataset):
    grokked_dataset = []

    d = dataset[0][0].shape[0]
    im_size = int(np.sqrt(d // 3))

    for sample in zip(dataset):
        image, label = sample[0]
        image = image.reshape(3, im_size, im_size)
        star_size = max(im_size // 20, 5)

        if label[0] == 1:
            star_color = -1
        if label[1] == 1:
            star_color = 1

        # Generate the star shape tensor
        x_pos = im_size - 2 * star_size
        y_pos = 2 * star_size
        center = star_size // 2
        right = torch.arange(center + 1, star_size)
        left = torch.arange(center)
        top = torch.arange(center - 1, -1, -1)
        bottom = torch.arange(star_size - 1, center, -1)
        star = torch.zeros((1, star_size, star_size))
        # Horizontal line
        star[:, center, :] = star_color
        # Vertical line
        star[:, :, center] = star_color
        # Bottom left diagonal
        star[:, right, left] = star_color
        # Top right diagonal
        star[:, left, right] = star_color
        # Top left diagonal
        star[:, left, top] = star_color
        # Bottom right diagonal
        star[:, right, bottom] = star_color

        # Expand and concatenate the star tensor to the center of the image tensor
        image[:, y_pos : y_pos + star_size, x_pos : x_pos + star_size] += star.expand(
            -1, -1, -1
        )
        image = torch.clamp(image, 0, 1)
        image = image.flatten()
        grokked_dataset.append((image, label))

    return grokked_dataset
