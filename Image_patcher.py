from torch import nn
import torch
from itertools import cycle
from torch.nn import functional as F
class ImagePatcher(nn.Module):
    def __init__(
            self, dim_hidden, input_feature_dims,model_image_n_patches,model_image_n_channels,model_image_n_classes):
        super(ImagePatcher, self).__init__()


        # Global setting, avoids non-patching logic in NPT init
        self.model_image_patching = True

        D = input_feature_dims  # Flattened size of an image

        # We reduce what the core model sees to a sequence of patches
        # (unordered, but with patch index embeddings), until the decoder
        self.image_n_patches = model_image_n_patches

        # Includes the target column
        self.num_input_features = self.image_n_patches + 1

        # Options: {'linear'}
        self.image_patch_type = 'linear'


        # e.g. 3 for RGB
        self.image_n_channels = model_image_n_channels

        # If we use BERT augmentation, this must be 2, for
        # the continuous pixel intensity and the mask value.
        # Otherwise, this should be 1.
        self.dim_intensity = 2
        self.dim_target_col = model_image_n_classes + 1
        # Exclude target column (we assume it is right concatenated)
        image_input_shape = (D - 1, self.dim_intensity)

        # The number of patches must divide the number of pixels
        assert image_input_shape[0] % self.image_n_patches == 0

        # This is in raw intensities, i.e. counting each pixel in an
        # RGB image thrice
        self.patch_size = image_input_shape[0] // self.image_n_patches

        # Compute resizing constants
        n_features = input_feature_dims - 1
        assert n_features % self.image_n_channels == 0

        # H = height, note that we are expecting square images for now
        # H = height = W = width
        flattened_image_size = n_features // self.image_n_channels
        self.image_H = int(flattened_image_size ** 0.5)
        assert flattened_image_size // self.image_H == self.image_H

        # Get number of rows of patches
        n_patches_per_side = self.image_n_patches ** 0.5
        assert int(n_patches_per_side) == n_patches_per_side
        n_patches_per_side = int(n_patches_per_side)

        # Get length of patches
        # (i.e. each patch is patch_side_length x patch_side_length)
        assert self.image_H % n_patches_per_side == 0
        self.patch_side_length = self.image_H // n_patches_per_side

        # ### Embeddings ###

        # Always use a linear out-embedding
            # Output into the number of intensities in a patch
            # (no mask dim needed), applied in a sliding fashion
        self.out_feature_embedding = nn.ModuleList([
            nn.Linear(dim_hidden, self.patch_size)])


        self.out_target_embedding = nn.Linear(
            dim_hidden, model_image_n_classes)

    def decode(self, X):
        # We receive a tensor of shape (N, n_patches + 1, E)

        # Feature Patch De-Embedding
        de_embeds = cycle(self.out_feature_embedding)


        X_ragged = {}

        # Projects each batched feature patch of shape (N, E) to (N,
        counter = 0
        for patch_index in range(X.shape[1] - 1):
            # X_patch.shape = (N, E)
            X_patch = X[:, patch_index, :]

            # de_embed.shape = (E, p) where p = patch size
            de_embed = next(de_embeds)

            # X_de_embed.shape = (N, p)
            X_de_embed = de_embed(X_patch)

            # Split into p columns of shape (N, 1)
            X_de_embed = torch.split(X_de_embed, 1, dim=1)
            for i in range(self.patch_size):
                X_ragged[counter +i] = X_de_embed[i]
            counter += self.patch_size
            #X_ragged += X_de_embed

        # Append projection of target column
        X_ragged[counter] = self.out_target_embedding(X[:, -1, :])

        return X_ragged

    def preprocess_flattened_image(self, X_ragged):
        """
        Prior to applying the Linear transforms, we wish to reshape
        our features, which constitute the image:
            * D = total number of columns (including the target)
            (N, D - 1, dim_intensity)
            where dim_intensity is 2 if we are using masking, 1 otherwise
            to (N, (D - 1) // n_channels, dim_intensity * n_channels)

        This is necessary because, e.g., CIFAR-10 flattens images to be of
            format 1024 R, 1024 G, 1024 B. We must reshape to make sure
            the patching has the correct receptive fields.

        Returns:
            Reshaped X_features, X_target column
        """
        # Shape (N, D - 1, dim_intensity)
        # where dim_intensity = 2 if we have continuous pixel intensity + mask
        # or 1 if we just have the pixel intensity (no BERT augmentation mask)
        #X_features = torch.stack(X_ragged[:-1], 1)
        X_features = X_ragged[:,:-1]
        # Reshape to (N, (D - 1) // n_channels, dim_intensity * n_channels)
        X_features = torch.reshape(
            X_features,
            (X_features.size(0),
             X_features.size(1) // self.image_n_channels,
             self.dim_intensity * self.image_n_channels))

        # Shape (N, 1, H_j) where H_j = num_categories + bool(is_mask)
        # (e.g. 2, for image regression with BERT augmentation)
        X_target = X_ragged[:,-1]

        return X_features, X_target


class LinearImagePatcher(ImagePatcher):
    def __init__(self, encoded_data,n_patches,device):
        dim_hidden = encoded_data.data.embedding_dim
        input_feature_dims = encoded_data.data.orig_dim
        image_channels = encoded_data.data.image_channels
        self.n_classes = encoded_data.data.n_classes
        self.device = device
        super(LinearImagePatcher, self).__init__(
            dim_hidden, input_feature_dims,n_patches,image_channels,self.n_classes)

        self.patch_n_pixels = self.patch_side_length * self.patch_side_length
        pixel_input_dims = self.dim_intensity * self.image_n_channels

        # Each patch embedding should be shape
        # (patch_n_pixels, (1 + bool(is_mask)) * n_channels, dim_feature_embedding)
        self.in_feature_embedding = nn.ParameterList([
            nn.Parameter(torch.empty(
                self.patch_n_pixels, pixel_input_dims,
                dim_hidden))])



        for embed in self.in_feature_embedding:
            nn.init.xavier_uniform_(embed)

        self.in_target_embedding = nn.Linear(
            self.dim_target_col, dim_hidden)
        self.to(device)

    def encode(self, X_ragged):
        # Feature Patch Embedding
        # Embed to a list of n_patch tensors,
        # each of size (N, dim_feature_embedding)

        X_features, X_target = self.preprocess_flattened_image(X_ragged)

        embeds = cycle(self.in_feature_embedding)


        X_embeds = {}
        counter = 0
        for pixel_index in range(0, X_features.shape[1], self.patch_n_pixels):
            # Projection:
            # n: batch dimension, number of rows
            # p: patch size in number of locations (e.g., num RGB pixels)
            # h: dim_intensity * n_channels
            #       = (1 + 1) * n_channels if we use BERT masking,
            #       = 1 * n_channels otherwise
            # e: dim_feature_embedding, NPT hidden dimensions

            # X_input.shape = (n, p, h)
            X_input = X_features[
                :, pixel_index:pixel_index+self.patch_n_pixels, :]

            # embed.shape = (p, h, e)
            embed = next(embeds)

            X_embeds[counter] = (torch.einsum('nph,phe->ne', X_input, embed))
            counter += 1
        encoded_col = torch.zeros((X_target.shape[0], self.n_classes)).long().to(self.device)
        ind = X_target[:,0] != -1
        to_encode = X_target[ind,0].long()
        if to_encode.nelement() != 0:
             encoded_col[ind] = F.one_hot(to_encode, num_classes=self.n_classes)
        X_target = torch.cat((encoded_col,X_target[:,1].reshape(-1,1)),dim=1)
        X_embeds[counter] = self.in_target_embedding(X_target)
        #X_embed = torch.stack(X_embeds, 1)

        return X_embeds




