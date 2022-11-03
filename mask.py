import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn


def get_mask(img):
    """ Performs object segmentation on the image and computes the boolean mask.

        # Parameters:
           @img, the image, to be segemented.


       # Returns the object mask.
       """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    img_list = [img]

    images = [transforms(d) for d in img_list]

    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    output = model(images)

    score_threshold = .75
    proba_threshold = 0.5

    boolean_masks = [
        out['masks'][out['scores'] > score_threshold] > proba_threshold
        for out in output
    ]

    return boolean_masks[0][0].permute(0, 2, 1)[0]


def applyMaterial(mask, original, new):
    """ Applies material from the new image to the original one, based on the mask.

            # Parameters:
               @mask, the boolean mask of the object.
               @original, the orginal image.
               @new, the new image with the trasferred material.


           # Returns the combination of the original image background and the object with the new material.
           """
    res = np.zeros((512, 512, 3))
    original = torch.permute(original, (1, 2, 0))
    new = new.cpu().clone()
    new = new.squeeze(0)
    new = torch.permute(new, (1, 2, 0)).detach().numpy()
    # iterate over mask pixels
    for i, row in enumerate(mask):
        for j, ele in enumerate(row):
            if ele:
                res[j][i] = new[j][i]
            else:
                res[j][i] = 255 - original.data[j][i]
    return res
