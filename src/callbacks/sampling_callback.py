import torch
import lightning as L
from matplotlib import pyplot as plt
from ..models.utils.ddpm_utils import get_tensor_image, get_index_from_list
import os


class SamplingCallback(L.Callback):
    '''
    samples noise and uses trained model to predict less noisy images
    '''

    def __init__(self,img_size:int, out:str) -> None:
        super().__init__()
        self.img_size = img_size
        self.out = out


    def on_train_epoch_end(self, trainer: L.Trainer, pl_module:L.LightningModule) -> None:
        # do something with all training_step outputs, for example:
        self.sample_plot_image(pl_module)


    @torch.no_grad()
    def sample_timestep(self, x:torch.Tensor, t:torch.Tensor, **kwargs) -> None:
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        betas = kwargs['betas']
        device = kwargs['device']
        sqrt_one_minus_alphas_cumprod  = kwargs['sqrt_one_minus_alphas_cumprod']
        sqrt_recip_alphas = kwargs['sqrt_recip_alphas']
        net = kwargs['net']
        posterior_variance = kwargs['posterior_variance']

        betas_t = get_index_from_list(betas, t, x.shape,device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            sqrt_one_minus_alphas_cumprod, t, x.shape, device
        )
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape,device)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * net(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape,device)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def sample_plot_image(self, pl_module:L.LightningModule) -> None:
        # Sample noise
        img = torch.randn((1, 3, self.img_size, self.img_size), device=pl_module.device)
        plt.figure(figsize=(15,15))
        plt.axis('off')
        num_images = 10
        stepsize = int(pl_module.T/num_images)

        plmodule_data = dict(betas=pl_module.betas,device=pl_module.device,\
                             sqrt_one_minus_alphas_cumprod=pl_module.sqrt_one_minus_alphas_cumprod,\
                                sqrt_recip_alphas=pl_module.sqrt_recip_alphas,net=pl_module.net,\
                                    posterior_variance=pl_module.posterior_variance)

        for i in range(0,pl_module.T)[::-1]:
            t = torch.full((1,), i, device=pl_module.device, dtype=torch.long)
            img = self.sample_timestep(img, t, **plmodule_data)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                plt.subplot(1, num_images, int(i/stepsize)+1)
                less_noisy_img = get_tensor_image(img.detach().cpu())
                plt.imshow(less_noisy_img)


        plt.savefig(os.path.join(self.out,f'epoch-{pl_module.current_epoch}.png'),dpi=200,bbox_inches="tight")
        plt.close()
