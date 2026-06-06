import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, gamma=0.01, epsilon=0.001):
        """
        Physics-Informed Loss Engine for High-Resolution Atmospheric Motion Vector Retrieval.

        Args:
            alpha (float): Regularization weight for Fluid Smoothness (SC).
            beta (float): Optimization weight for Constancy Gradient (GC).
            gamma (float): Baseline anchor weight for Hinting Background (E_W).
            epsilon (float): Structural outlier dampening constant.
        """
        super(PhysicsInformedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def warp(self, img, flow):
        """
        Executes backward warping via differentiable bilinear interpolation to calculate I(x+U).
        Maps Frame t+1 features back into the tracking frame coordinates of Frame t.
        """
        B, C, H, W = img.size()

        # Build 2D coordinate space meshgrid
        yy, xx = torch.meshgrid(
            torch.arange(0, H, device=img.device, dtype=torch.float32),
            torch.arange(0, W, device=img.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # Shape: [B, 2, H, W]
        vgrid = grid + flow # Apply spatial vector displacement field

        # Standardize matrix coordinates into PyTorch grid range [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1) # Shift shape configuration to [B, H, W, 2]

        return F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border', align_corners=True)

    def compute_spatial_gradients(self, tensor):
        """Calculates discrete spatial differences across dimensions (X-axis and Y-axis)."""
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

        # Replicate borders to keep spatial sizing dimensions uniform
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        return dx, dy

    def apply_charbonnier(self, residual_tensor):
        """Wraps mathematical tracking outliers with the Charbonnier threshold formula."""
        return torch.sqrt(residual_tensor ** 2 + self.epsilon ** 2)

    def forward(self, flow_pred, height_pred, img1, img2, background_flow=None):
        """
        Evaluates fluid transport parameters against consecutive multi-spectral images.
        """
        # TODO: scientific validation required
        # Note: height_pred (cloud-top pressure prediction) is currently completely ignored in the loss formulation,
        # meaning the height prediction head is untrained. This will be addressed in a future scientific update.
        
        # 1. Warp source tracking field back to baseline frame reference
        img2_warped = self.warp(img2, flow_pred)

        # 2. Brightness Constancy Loss (BC)
        bc_residual = img2_warped - img1
        loss_bc = torch.mean(self.apply_charbonnier(bc_residual))

        # 3. Constancy Gradient Loss (GC)
        img1_dx, img1_dy = self.compute_spatial_gradients(img1)
        warped_dx, warped_dy = self.compute_spatial_gradients(img2_warped)
        gc_residual_x = warped_dx - img1_dx
        gc_residual_y = warped_dy - img1_dy
        loss_gc = torch.mean(self.apply_charbonnier(gc_residual_x) + self.apply_charbonnier(gc_residual_y))

        # 4. Fluid Smoothness Loss (SC)
        u_channel, v_channel = flow_pred[:, 0:1, :, :], flow_pred[:, 1:2, :, :]
        u_dx, u_dy = self.compute_spatial_gradients(u_channel)
        v_dx, v_dy = self.compute_spatial_gradients(v_channel)
        loss_sc = torch.mean(
            self.apply_charbonnier(u_dx) + self.apply_charbonnier(u_dy) +
            self.apply_charbonnier(v_dx) + self.apply_charbonnier(v_dy)
        )

        # 5. Hinting Background Loss (E_W)
        if background_flow is not None:
            bg_residual = flow_pred - background_flow
            loss_ew = torch.mean(self.apply_charbonnier(bg_residual))
        else:
            loss_ew = torch.tensor(0.0, device=flow_pred.device)

        # 6. System Aggregation using Hyperparameter Coefficients
        total_loss = loss_bc + (self.beta * loss_gc) + (self.alpha * loss_sc) + (self.gamma * loss_ew)

        # Isolate visual tracking data metrics from fluid physics metrics for the console dashboard
        data_tracking_metric = loss_bc + (self.beta * loss_gc)
        fluid_smoothness_metric = loss_sc

        return total_loss, data_tracking_metric, fluid_smoothness_metric
