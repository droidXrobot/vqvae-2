import torch
import hydra
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms as transforms
from vqvae2 import VQVAE, VQVAE2
from torchvision.utils import save_image


def encode_decode_single_image(image_path, checkpoint_path, cfg, device='cuda'):
    """
    Given an image path and the path to a trained VQ-VAE-2 checkpoint,
    this function will encode and then decode the image, outputting
    both the resulting image and the latent encoding.
    """
    # Load image using any library of your choice (PIL, OpenCV, etc.)
    # This code assumes you're using PIL
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Load the VQ-VAE-2 model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net = VQVAE2.build_from_config(
        cfg.vqvae.model, 
        codebook_gumbel_temperature=0.1, 
        codebook_init_type="kaiming_uniform", 
        codebook_cosine=True
    )
    net.load_state_dict(checkpoint)
    net = net.to(device)
    
    # Encode and decode the image
    with torch.no_grad():
        net.eval()
        recon, _, _, _ = net(image)
    
    # We detach and move to cpu for further visualization
    recon = recon.detach().cpu()

    # Latent encoding is given by the `indices` returned by `net`
    _, _, indices, _ = net(image)
    
    return recon, indices

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Replace these paths with the ones you're using
    # Call the training function to train the model
    # After training, use the model to encode and decode an image
    # For example, if your image is at './image.jpg' and your model checkpoint is at './model.pt'
    image_path = '/content/00056.png'
    checkpoint_path = '/content/state_dict_final.pt'
    
    recon_image, latent_codes = encode_decode_single_image(image_path, checkpoint_path, cfg)

    # Save the reconstructed image
    save_image(recon_image, 'reconstructed_image.png')

if __name__ == "__main__":
    main()
