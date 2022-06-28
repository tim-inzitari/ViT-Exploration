import torch
from torch import nn
import numpy as np
from MSA import MSA

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/(10000 ** (j / d))) if j % 2 == 0 else np.cos(i/ (10000 ** ((j-1) / d)))
    return result




class ViT(nn.Module):
    def __init__(self, input_shape, n_patches=7, hidden_d = 8, n_heads=2, out_d=10):
        super(ViT, self).__init__()

        #input and patches
        self.input_shape = input_shape #input shape is w.r.t images -> (N, C, H, W)
        self.n_patches = n_patches
        self.patch_size = (input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.hidden_d = hidden_d

        # sanity check
        assert input_shape[1] % n_patches == 0, "input shape not entirely divisible by number of patches (dim 1)"
        assert input_shape[2] % n_patches == 0, "input shape not entirely divisible by number of patches (dim 2)"

        #1 lin map
        self.input_d = int(input_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.lin_mapper = nn.Linear(self.input_d, self.hidden_d)

        #2 class token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        #3 positional embeddings
        # in forward method

        #4 layer norm 1 and MSA on classification token
        self.ln1 = torch.nn.LayerNorm((self.n_patches ** 2 + 1, self.hidden_d))
        self.msa = MSA(self.hidden_d, n_heads)

        #5 layer norm 2 and encoder MLP
        self.ln2 = nn.LayerNorm((self.n_patches**2 + 1, self.hidden_d))
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU()
        )

        #6 classifier mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, images):

        #div image to patches
        n, c, w, h = images.shape
        patches = images.reshape(n, self.n_patches ** 2, self.input_d)

        #run lin map for tokenizer
        tokens = self.lin_mapper(patches)


        # add classification token to tokesn
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        #add positional embeddings
        tokens += get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d).repeat(n,1,1)


        # Transformer ENcoding begins
        #multiple encoder blocks together
        # layer norm, MSA, and residual connection

        out = tokens + self.msa(self.ln1(tokens))


        #layer norm mlp and residual cons
        out = out + self.enc_mlp(self.ln2(out))

        #transformer encoder ends

        #get classification token only
        out = out[:, 0]

        return self.mlp(out)

if __name__ == '__main__':
    model = ViT (
        input_shape=(1,28,28),
        n_patches=7
    )
    x = torch.rand(3, 1, 28, 28) # fake image
    print(model(x).shape) 

    import matplotlib.pyplot as plt

    plt.imshow(get_positional_embeddings(100,300), cmap='hot', interpolation='nearest')
    plt.show