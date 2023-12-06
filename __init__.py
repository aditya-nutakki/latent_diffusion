"""


https://arxiv.org/pdf/2112.10752.pdf

By decomposing the image formation process into a se-
quential application of denoising autoencoders, diffusion
models (DMs) achieve state-of-the-art synthesis results on
image data and beyond. Additionally, their formulation al-
lows for a guiding mechanism to control the image gen-
eration process without retraining. However, since these
models typically operate directly in pixel space, optimiza-
tion of powerful DMs often consumes hundreds of GPU
days and inference is expensive due to sequential evalu-
ations. To enable DM training on limited computational
resources while retaining their quality and flexibility, we
apply them in the latent space of powerful pretrained au-
toencoders. In contrast to previous work, training diffusion
models on such a representation allows for the first time
to reach a near-optimal point between complexity reduc-
tion and detail preservation, greatly boosting visual fidelity.
By introducing cross-attention layers into the model archi-
tecture, we turn diffusion models into powerful and flexi-
ble generators for general conditioning inputs such as text
or bounding boxes and high-resolution synthesis becomes
possible in a convolutional manner. Our latent diffusion
models (LDMs) achieve new state-of-the-art scores for im-
age inpainting and class-conditional image synthesis and
highly competitive performance on various tasks, includ-
ing text-to-image synthesis, unconditional image generation
and super-resolution, while significantly reducing computa-
tional requirements compared to pixel-based DMs.


"""
