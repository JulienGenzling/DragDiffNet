import os
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as plu
from mesh_utils.mesh import TriMesh

from tqdm import tqdm 

class Trainer(object):

    def __init__(self, diffusionnet_cls, model_cfg, train_loader, valid_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, num_epochs=200,
                 lr_decay_every = 50, lr_decay_rate = 0.5,
                 log_interval=10, save_dir=None):

        """
        diffusionnet_cls: (nn.Module) class of the DiffusionNet model
        model_cfg: (dict) keyword arguments for model
        train_loader: (torch.utils.DataLoader) DataLoader for training set
        valid_loader: (torch.utils.DataLoader) DataLoader for validation set
        device: (str) 'cuda' or 'cpu'
        lr: (float) learning rate
        weight_decay: (float) weight decay for optimiser
        num_epochs: (int) number of epochs
        lr_decay_every: (int) decay learning rate every this many epochs
        lr_decay_rate: (float) decay learning rate by this factor
        log_interval: (int) print training stats every this many iterations
        save_dir: (str) directory to save model checkpoints
        """

        # TOD build the network from the model_cfg
        self.model = diffusionnet_cls(
            p_in=model_cfg['p_in'],
            p_out=model_cfg['p_out'],
            n_channels=model_cfg['n_channels'],
            N_block=model_cfg['N_block']
        )


        self.loss = nn.MSELoss()



        ## THIS PART JUST STORES SOME OTHER PARAMETERS
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        self.log_interval = log_interval
        self.save_dir = save_dir

        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

        self.inp_feat = model_cfg.get('inp_feat', 'xyz')
        self.num_eig = model_cfg.get('num_eig', 128)
        if not self.inp_feat in ['xyz', 'hks', 'wks']:
            raise ValueError('inp_feat must be one of xyz, hks, wks')

        self.model.to(self.device)


    def forward_step(self, verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY):
        """
        Perform a forward step of the model.

        Args:
            verts (torch.Tensor): (N, 3) tensor of vertex positions
            faces (torch.Tensor): (F, 3) tensor of face indices
            frames (torch.Tensor): (N, 3, 3) tensor of tangent frames.
            vertex_area (torch.Tensor): (N, N) sparse Tensor of vertex areas.
            L (torch.Tensor): (N, N) sparse Tensor of cotangent Laplacian.
            evals (torch.Tensor): (num_eig,) tensor of eigenvalues.
            evecs (torch.Tensor): (N, num_eig) tensor of eigenvectors.
            gradX (torch.Tensor): (N, N) tensor of gradient in X direction.
            gradY (torch.Tensor): (N, N) tensor of gradient in Y direction.

        Returns:
            pred (torch.Tensor): (N, p_out) tensor of predicted labels.
        """

        if self.inp_feat == 'xyz':
            features = verts
        elif self.inp_feat == 'hks':
            features = self.compute_HKS(verts, faces, self.num_eig, n_feat=32)
        elif self.inp_feat == 'wks':
            features = self.compute_WKS(verts, faces, self.num_eig, num_E=32)

        preds = self.model(features, vertex_area, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        # MAYBE ADD ACTIVATION
        return preds



    def train_epoch(self):
        """
        Train the network for one epoch
        """
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(tqdm(self.train_loader, "Train epoch")):

            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            label = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            # MAYBE DO SOMETHING TO THE PREDS

            # COMPUTE THE LOSS
            print(preds, label)
            loss = self.loss(preds, label)##

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # COMPUTE TRAINING ACCURACY
            pred_labels = torch.argmax(preds, dim=-1)# TODO GET PREDICTED LABELS

            n_correct = pred_labels.eq(label).sum().item() # number of correct predictions
            train_acc += n_correct/label.shape[0]

        return train_loss/len(self.train_loader), train_acc/len(self.train_loader)

    def valid_epoch(self):
        """
        Run a validation epoch
        """
        val_loss = 0
        val_acc = 0
        print("Start val epoch")
        for i, batch in enumerate(self.valid_loader):

            # READ BATCH
            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            labels = batch["labels"].to(self.device)

            # TODO PERFORM FORWARD STEP
            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            # MAYBE DO SOMETHING TO THE PREDS

            # Compute Loss - THIS DEPENDS ON YOUR CHOICE OF LOSS
            loss = self.loss(preds, labels)##

            val_loss += loss.item()

            # Compute ACCURACCY
            pred_labels = torch.argmax(preds, dim=-1)# TODO GET PREDICTED LABELS

            n_correct = pred_labels.eq(labels).sum().item() # number of correct predictions
            val_acc += n_correct/labels.shape[0]
        print("End val epoch")
        return val_loss/len(self.valid_loader), val_acc/len(self.valid_loader)

    def run(self):
        os.makedirs('./models', exist_ok=True)
        for epoch in range(self.num_epochs):
            self.model.train()

            if epoch % self.lr_decay_every == 0:
                self.adjust_lr()

            train_ep_loss, train_ep_acc = self.train_epoch()
            self.train_losses.append(train_ep_loss)
            self.train_accs.append(train_ep_acc)

            if epoch % self.log_interval == 0:
                val_loss, val_acc = self.valid_epoch()
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pth'))
                print(f'Epoch: {epoch:03d}/{self.num_epochs}, '
                      f'Train Loss: {train_ep_loss:.4f}, '
                      f'Train Acc: {1e2*train_ep_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {1e2*val_acc:.2f}%')
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_final.pth'))


    def visualize(self):
        """
        We only test the first two shapes of validation set.
        """
        self.model.eval()
        test_seg_meshes = []

        for i, batch in enumerate(self.valid_loader):
            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            labels = batch["labels"].to(self.device)


            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            pred_labels = torch.max(preds, dim=1).indices

            test_seg_meshes.append([TriMesh(verts.cpu().numpy(), faces.cpu().numpy()),
                                  pred_labels.cpu().numpy()])
            if i==1:
                break


        cmap1 = plt.get_cmap("jet")(test_seg_meshes[0][-1] / (146))[:,:3]
        cmap2 = plt.get_cmap("jet")(test_seg_meshes[1][-1] / (146))[:,:3]

        plu.double_plot(test_seg_meshes[0][0], test_seg_meshes[1][0], cmap1, cmap2)
        #return plot_multi_meshes(test_seg_meshes, cmap='vert_colors')

    def adjust_lr(self):
        lr = self.lr * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_HKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the HKS features for each vertex in the mesh.
        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            hks (torch.Tensor): (N, n_feat) tensor of HKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig]

        t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], n_feat)
        t_list = torch.from_tensor(t_list.astype(np.float32)).to(device=evecs.device)

        evals_s = abs_ev

        coefs = torch.exp(-t_list[:,None] * evals_s[None,:])  # (num_T,K)

        natural_HKS = np.einsum('tk,nk->nt', coefs, evecs[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_T)

        return (1/inv_scaling)[None,:] * natural_HKS

    def compute_WKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the WKS features for each vertex in the mesh.

        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            wks: torch.Tensor: (N, num_E) tensor of WKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig]

        e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
        sigma = 7*(e_max-e_min)/n_feat

        e_min += 2*sigma
        e_max -= 2*sigma

        energy_list = torch.linspace(e_min,e_max,n_feat)

        evals_s = abs_ev

        coefs = torch.exp(-torch.square(energy_list[:,None] - torch.log(torch.abs(evals_s))[None,:])/(2*sigma**2))  # (num_E,K)

        natural_WKS = np.einsum('tk,nk->nt', coefs, evecs[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS

