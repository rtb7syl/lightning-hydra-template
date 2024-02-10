'''
import torch
import lightning as L
from matplotlib import pyplot as plt

class SamplingCallback(L.Callback):

    def __init__(self,img_size) -> None:
        super().__init__()
        self.img_size = img_size
    
    #samples noise and uses trained model to predict less noisy images
    
    def on_train_epoch_end(self, trainer, pl_module):
        # do something with all training_step outputs, for example:
        self.sample_plot_image(pl_module, ep=trainer.current_epoch,step=trainer.global_step)

    @torch.no_grad()
    def sample_timestep(pl_module, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = pl_module.get_index_from_list(pl_module.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pl_module.net(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample_plot_image(self,pl_module):
        # Sample noise
        img = torch.randn((1, 3, img_size, img_size), device=device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(T/num_images)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(model, img, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                less_noisy_img = show_tensor_image(img.detach().cpu())
                plt.imshow(less_noisy_img)


        plt.savefig(f'epoch-{ep}-step-{step}.png',dpi=200,bbox_inches="tight")
        plt.close()
'''