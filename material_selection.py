import glob

from style_transfer import *


def get_images(style):
    imgs = []
    files = glob.glob("data/dataset/dtd/dtd/images/" + style + "/*.jpg")
    for img in files:
        image = image_loader(img, device='cuda', img_size=512)
        imgs.append(image)

    return imgs

def find_style_image(content_img, style):
    content_layers = ['conv4_2']
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    # Set random seed for better reproducibility
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("we CUDA but we dont")

    vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    normed_content_img = normalize(content_img, vgg_mean, vgg_std)
    # Retrieve feature maps for content and style image
    content_features = model(normed_content_img)

    images = get_images(style)

    bestIm = None
    lowest_loss = 9999999
    for img in images:
        # Either initialize the image from random noise or from the content image
        optim_img = torch.nn.Parameter(img, requires_grad=True)

        # Retrieve features of image that is being optimized
        normed_img = normalize(optim_img, vgg_mean, vgg_std)
        input_features = model(normed_img)
        loss = content_loss(input_features, content_features, content_layers)
        if loss < lowest_loss:
            bestIm = img
            lowest_loss = loss
    return bestIm
