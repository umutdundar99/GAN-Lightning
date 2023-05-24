from torch import nn

def get_discriminator_block(input_dim:int, output_dim:int):
    return nn.Sequential(
         nn.Linear(input_dim, output_dim), 
         nn.LeakyReLU(0.2, inplace=True)
    )