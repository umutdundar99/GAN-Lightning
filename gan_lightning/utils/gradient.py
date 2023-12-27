import torch

def compute_gradient_penalty(discriminator: torch.nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, alpha: torch.Tensor):
    '''
    Calculate the penalty term for the gradient in Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Args:
        discriminator (torch.nn.Module): The discriminatoric neural network (discriminator).
        real_samples (torch.Tensor): A batch of genuine images.
        fake_samples (torch.Tensor): A batch of generated images.
        alpha (torch.Tensor): A vector specifying random blending coefficients for real_samples and fake_samples images.

    Returns:
        torch.Tensor: The computed gradient penalty term.

    Overview:
        This function is designed for WGAN-GP training to penalize the model when the discriminatoric's behavior deviates from Lipschitz continuity.

    Detailed Steps:
        1. Create Blended Images:
            - Generate mixed images by blending real_samples and fake_samples images based on random coefficients.
            - Formula: blended_images = real_samples * alpha + fake_samples * (1 - alpha)

        2. Evaluate discriminatoric Scores:
            - Pass the mixed images through the discriminatoric to obtain scores.
            - This step measures how closely the discriminatoric rates each mixed image to real_samples data.
            - Formula: mixed_scores = discriminator(blended_images)

        3. Derive Image Gradient:
            - Use torch.autograd.grad to compute the gradient of the discriminatoric's scores with respect to the mixed images.
            - The gradient signifies how variations in pixel values of mixed images impact discriminatoric scores.
            - Parameters:
                - inputs: blended_images
                - outputs: mixed_scores
                - grad_outputs: A tensor of ones with the same shape as mixed_scores, indicating the initial gradient.
                - create_graph: Construct a computation graph for potential higher-order gradients.
                - retain_graph: Retain the computation graph for potential subsequent backward passes.

        4. Reshape and Calculate Gradient Norm:
            - Reshape the gradient to have one row per sample in the batch.
            - Compute the L2 norm (Euclidean norm) of the reshaped gradient along dimension 1.
            - Formula: gradient_norm = torch.norm(gradient, 2, dim=1)

        5. Compute Penalty:
            - Calculate the penalty term as the mean squared difference between the gradient norm and 1.
            - Formula: penalty_gradient = torch.mean((gradient_norm - 1)**2)

        6. Return Penalty Term:
            - Provide the computed gradient penalty term for use in the overall loss function during WGAN-GP training.
    '''
    
    mixed_images = (real_samples * alpha) + (fake_samples * (1 - alpha))

    mixed_scores = discriminator(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    
    # Penalizes deviations from the target norm of 1. 
    # The model is encouraged to produce gradients with a norm close to 1 by minimizing this penalty term.
    penalty_gradient = torch.mean((gradient.norm(2, dim=1) - 1)**2)

    return penalty_gradient
