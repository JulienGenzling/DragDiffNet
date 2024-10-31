import torch 
import torch.nn as nn

def project_to_basis(x, evecs, vertex_areas):
    """
    Project an input sinal x to the spectral basis.


    Parameters
    -------------------
    x            : (B, n, p) Tensor of input
    evecs        : (B, n, K) Tensor of eigenvectors
    vertex_areas : (B, n,) vertex areax

    Output
    -------------------
    projected_values : (B, K, p) Tensor of coefficients in the basis
    """
    evecsT = evecs.transpose(-2, -1)
    a = x * vertex_areas.unsqueeze(-1)
    return evecsT @ a


def unproject_from_basis(coeffs, evecs):
    """
    Transform input coefficients in basis into a signal on the complete shape.

    Parameters
    -------------------
    coeffs : (B, K, p) Tensor of coefficients in the spectral basis
    evecs : (B, n, K) Tensor of eigenvectors

    Output
    -------------------
    decoded_values : (B, n, p) values on each vertex
    """
    return evecs @ coeffs


class SpectralDiffusion(nn.Module):

    def __init__(self, n_channels):
        """
        Initializes the module with time parameters to 0.

        Parameters
        ------------------
        n_channels : int - number of input feature functions
        """
        # This runs the __init__ function of nn.Module
        super().__init__()

        self.n_channels = n_channels

        ## TODO DEFINE AND INITIALIZE THE Diffusion times as learnable parameters.
        self.diffusion_times = nn.Parameter(torch.Tensor(n_channels))
        nn.init.constant_(self.diffusion_times, .0)


    def forward(self, x, evals, evecs, vertex_areas):
        """
        Given input features x and information on the current meshes
        return diffused versions of the features.

        Parameters
        ------------------------
        x     : (B, n, p) batch of input features. p = self.n_channels
        evals : (B, K,) batch of eigenvalues
        evecs : (B, n, K) batch of eigenvectors
        vertex_areas : (B, n,) batch of vertex areax


        Output
        ------------------------
        x_diffuse : diffused version of each input feature
        """
        # Remove negative diffusion times
        with torch.no_grad():
            self.diffusion_times.data = torch.clamp(self.diffusion_times, min=1e-8)

        ## TODO DIFFUSE x
        times = self.diffusion_times
        x_spec = project_to_basis(x, evecs, vertex_areas)
        x_diffuse_spec = torch.exp(-evals.unsqueeze(-1) * times.unsqueeze(0)) * x_spec 
        x_diffused = unproject_from_basis(x_diffuse_spec, evecs)
        
        return x_diffused

class SpatialGradient(nn.Module):
    """
    Module which computes g_v from vertex embeddings.
    """
    def __init__(self, n_channels):
        """
        Initializes the module.

        Parameters
        ------------------
        n_channels : int - number of input feature functions
        """

        super().__init__()

        self.n_channels = n_channels

        # Real and Imaginary part of B
        self.B_re = nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.B_im = nn.Linear(self.n_channels, self.n_channels, bias=False)

    def forward(self, vects):
        """
        Parameters
        ----------------------
        Vects : (N, P, 2) per-vertex vector field (w_v)

        Output
        ---------------------
        features : (N, P) per-vertex scalar field
        """
        vects_re = vects[...,0]  # (N,P) real part of w_v
        vects_im = vects[...,1]  # (N,P) imaginary part of w_v

        ## TODO Perform forward pass
        vectsB_re = self.B_re(vects_re)
        vectsB_im = self.B_im(vects_im)
        
        return torch.tanh(vects_re * vectsB_re + vects_im * vectsB_im)


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with activation and potential dropout
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU):
        """
        Activation and dropout is applied after all layer BUT the last one

        Parameters
        ---------------------------
        layer_size : list of ints - list of sizes of the MLP
        dropout    : book - whether to add droupout or not
        activation : nn.module : activation function
        """
        super().__init__()

        layer_list = []
        for i in range(1, len(layer_sizes)):
          if dropout:
            layer_list.append(nn.Dropout())
          layer_list.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

          if i<len(layer_sizes)-1:
            layer_list.append(activation())

        ## TODO FILL THE LAYER LIST
        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):
        """
        Parameters
        --------------------
        x : (n, p) - input features, batch size is the number of vertices !

        Output
        -------------------
        y : (n,p') - output features
        """
        # NOTHING TO DO HERE
        return self.layer(x)
    
class DiffusionNetBlock(nn.Module):
    """
    Complete Diffusion block
    """

    def __init__(self, n_channels, mlp_hidden_dims, dropout=True):
        """
        Initializes the module.

        Parameters
        ------------------
        n_channels      : int - number of feature functions (serves as both input and output)
        mlp_hidden_dims : list of int - sizes of HIDDEN layers of the miniMLP.
                          You should add the input and output dimension to it.
        """
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.n_channels = n_channels

        self.dropout = dropout

        # Diffusion block
        self.diff = SpectralDiffusion(n_channels)
        self.gradient_features = SpatialGradient(n_channels)
        mlp_hidden_dims = [3*n_channels, n_channels, n_channels, n_channels]
        self.mlp = MiniMLP(mlp_hidden_dims, dropout=dropout)


    def forward(self, x_in, vertex_areas, evals, evecs, gradX, gradY):
        """
        Parameters
        -------------------
        x_in         : (B,n,p) - Tensor of input signal.
        vertex_areas : (B,n) - Tensor of vertex areas
        evals        : (B, K,) batch of eigenvalues
        evecs        : (B, n, K) batch of eigenvectors
        gradX        : Half of gradient matrix, sparse real tensor with dimension [B,N,N]
        gradY        : Half of gradient matrix, sparse real tensor with dimension [B,N,N]

        Output
        -------------------
        x_out : (B,n,p) - Tensor of output signal.
        """

        # Manage dimensions
        B = x_in.shape[0] # batch dimension

        # Diffusion block
        x_diffuse = self.diff(x_in, evals, evecs, vertex_areas) # DIFFUSED X_in  # (B, N, p)


        # Compute the batch of gradients
        x_grads = [] # Manually loop over the batch
        for b in range(B):
            # gradient after diffusion
            x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
            x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

            x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))

        x_grad = torch.stack(x_grads, dim=0)  # (B, N, P, 2)

        # TODO EVALUATE GRADIENT FEATURES
        x_grad_features = self.gradient_features(x_grad) 

        # TODO APPLY THE MLP TO THE CONCATENATED FEATURES
        feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        x0_out = self.mlp(feature_combined)

        # TODO APPLY THE RESIDUAL CONNECTION
        x0_out = x0_out + x_in

        return x0_out

class DiffusionNet(nn.Module):

    def __init__(self, p_in, p_out, n_channels=128, N_block=4, last_activation=None, mlp_hidden_dims=None, dropout=True):
        """
        Construct a DiffusionNet.
        Parameters
        --------------------
        p_in            : int - input dimension of the network
        p_out           : int - output dimension  of the network
        n_channels      : int - dimension of internal DiffusionNet blocks (default: 128)
        N_block         : int - number of DiffusionNet blocks (default: 4)
        last_activation : int - a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax
        mlp_hidden_dims : list of int - a list of hidden layer sizes for MLPs (default: [C_width, C_width])
        dropout         : bool - if True, internal MLPs use dropout (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.p_in = p_in
        self.p_out = p_out
        self.n_channels = n_channels
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [n_channels, n_channels]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout


        ## TODO SETUP THE NETWORK (LINEAR LAYERS + BLOCKS)

        self.blocks = [] # TOFILL
        self.blocks.append(nn.Linear(self.p_in, self.n_channels))
        for i in range(N_block):
          self.blocks.append(DiffusionNetBlock(n_channels, mlp_hidden_dims, dropout))
        self.blocks.append(nn.Linear(self.n_channels, self.p_out))

        self.net = nn.ModuleList(self.blocks)

    def forward(self, x_in, vertex_areas, evals=None, evecs=None, gradX=None, gradY=None):
        """
        Progapate a signal through the network.
        Can handle input without batch dimension (will add a dummy dimension to set batch size to 1)

        Parameters
        --------------------
        x_in         : (n,p) or (B,n,p) - Tensor of input signal.
        vertex_areas : (n,) or (B,n) - Tensor of vertex areas
        evals        : (B, K,) or (K,) batch of eigenvalues
        evecs        : (B, n, K) or (n, K) batch of eigenvectors
        gradX        : Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        gradY        : Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Output
        -----------------------
        x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.p_in:
            raise ValueError(f"DiffusionNet was constructed with p_in={self.p_in}, "
                             f"but x_in has last dim={x_in.shape[-1]}")
        N = x_in.shape[-2]

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0) # (B, N, P)
            vertex_areas = vertex_areas.unsqueeze(0) # (B, N)
            if evals != None: evals = evals.unsqueeze(0) # (B,K)
            if evecs != None: evecs = evecs.unsqueeze(0) # (B,N,K)
            if gradX != None: gradX = gradX.unsqueeze(0) # (B,N,N)
            if gradY != None: gradY = gradY.unsqueeze(0) # (B,N,N)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else: raise ValueError("x_in should be tensor with shape (n,p) or (B,n,p)")

        ##  TODO PROCESS THE INPUTS
        x_p = self.blocks[0](x_in)
        for i in range(self.N_block):
          x_p = self.blocks[i+1](x_p, vertex_areas, evals, evecs, gradX, gradY)
        x_out = self.blocks[-1](x_p)
        
        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0) # (N, p_out)

        return x_out