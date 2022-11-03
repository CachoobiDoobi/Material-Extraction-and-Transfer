from pathlib import Path

import numpy as np
import torch.optim as optim
from torchvision.io import read_image

from helper_functions import *
from mask import get_mask, applyMaterial

# Choose what feature maps to extract for the content and style loss
# We use the ones as mentioned in Gatys et al. 2016
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

torch.manual_seed(2022)  # Set random seed for better reproducibility
device = 'cpu'
if torch.cuda.is_available():

    device = 'cuda'
else:
    print("we CUDA but we dont")

# Hyperparameters
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
img_size = 512
# Sets of hyperparameters that worked well for us
if img_size == 128:
    num_steps = 1000
    w_style_1 = 1e5
    w_style_2 = 1e5
    w_content = 1
    w_tv = 5e-4
else:
    num_steps = 1000
    w_style_1 = 1e6
    w_style_2 = 1e6
    w_content = 1
    w_tv = 5e-6


def normalize(img, mean, std):
    """ Normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.
    return transforms.Normalize(mean, std)(img)


def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.
    
    # Returns the content loss, a torch.tensor of size (1)
    """
    # TODO: 2. Implement the content loss given the input feature volume and the
    # content feature volume. Note that:
    # - Only the layers given in content_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.

    loss = 0.0
    for layer in content_layers:
        loss += torch.mean((content_features[layer].detach() - input_features[layer]) ** 2)
    loss /= len(content_layers)
    return loss


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.
    
    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix. 
    # Do not use for-loops, make use of Pytorch functionalities.
    (bs, ch, h, w) = x.size()
    f = x.view(bs * ch, w * h)
    G = torch.mm(f, f.t()) / (ch * h * w)

    return G


def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.
    
    # Returns the style loss, a torch.tensor of size (1)
    """
    # TODO: 3.1 Implement the style loss given the input feature volume and the
    # style feature volume. Note that:
    # - Only the layers given in style_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    # - Implement the gram_matrix function.
    loss = 0.0
    for layer in style_layers:
        input_gram = gram_matrix(input_features[layer])
        style_gram = gram_matrix(style_features[layer].detach())
        loss += torch.mean((style_gram - input_gram) ** 2)
    loss /= len(style_layers)
    return loss


def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)
    
    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss.
    tv_h = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :]).sum()
    tv_w = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1]).sum()

    # not normalizing gives better results
    return tv_h + tv_w


def run_single_image(vgg_mean, vgg_std, content_img, style_img, num_steps=num_steps,
                     random_init=True, w_style=w_style_1, w_content=w_content, w_tv=w_tv):
    """ Neural Style Transfer optmization procedure for a single style image.

    # Parameters:
        @vgg_mean, VGG channel-wise mean, torch.tensor of size (c)
        @vgg_std, VGG channel-wise standard deviation, detorch.tensor of size (c)
        @content_img, torch.tensor of size (1, c, h, w)
        @style_img, torch.tensor of size (1, c, h, w)
        @num_steps, int, iteration steps
        @random_init, bool, whether to start optimizing with based on a random image. If false,
            the content image is as initialization.
        @w_style, float, weight for style loss
        @w_content, float, weight for content loss
        @w_tv, float, weight for total variation loss

    # Returns the style-transferred image
    """

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # TODO: 1. Normalize Input images
    normed_style_img = normalize(style_img, vgg_mean, vgg_std)
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    # Retrieve feature maps for content and style image
    style_features = model(normed_style_img)
    content_features = model(normed_content_img)

    # Either initialize the image from random noise or from the content image
    if random_init:
        optim_img = torch.randn(content_img.data.size(), device=device)
        optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
    else:
        optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

    # Initialize optimizer and set image as parameter to be optimized
    optimizer = optim.LBFGS([optim_img])

    # Training Loop
    iter = [0]
    while iter[0] <= num_steps:

        def closure():

            # Set gradients to zero before next optimization step
            optimizer.zero_grad()

            # Clamp image to lie in correct range
            with torch.no_grad():
                optim_img.clamp_(0, 1)

            # Retrieve features of image that is being optimized
            normed_img = normalize(optim_img, vgg_mean, vgg_std)
            input_features = model(normed_img)

            # TODO: 2. Calculate the content loss
            if w_content > 0:
                c_loss = w_content * content_loss(input_features, content_features, content_layers)
            else:
                c_loss = torch.tensor([0]).to(device)

            # TODO: 3. Calculate the style loss
            if w_style > 0:
                s_loss = w_style * style_loss(input_features, style_features, style_layers)
            else:
                s_loss = torch.tensor([0]).to(device)

            # TODO: 4. Calculate the total variation loss
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)
            else:
                tv_loss = torch.tensor([0]).to(device)

            # Sum up the losses and do a backward pass
            loss = s_loss + c_loss + tv_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % 50 == 0:
                print('iter {}: | Style Loss: {:4f} | Content Loss: {:4f} | TV Loss: {:4f}'.format(
                    iter[0], s_loss.item(), c_loss.item(), tv_loss.item()))

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)

    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    return optim_img


def run_material_transfer(content_img_path, style_img_1):
    # Paths
    out_folder = 'outputs'
    content_img_path = content_img_path

    content_img = image_loader(content_img_path, device=device, img_size=img_size)
    # Define the channel-wise mean and standard deviation used for VGG training

    # Single image optimization
    print('Start single style image optimization.')
    output1 = run_single_image(
        vgg_mean, vgg_std, content_img, style_img_1, num_steps=num_steps,
        random_init=False, w_style=w_style_1, w_content=w_content, w_tv=w_tv)
    output_name1 = f'single img_size-{img_size} num_steps-{num_steps} w_style-{w_style_1} w_content-{w_content} w_tv-{w_tv}'
    save_image(output1, title=output_name1, out_folder=out_folder)
    return output1
