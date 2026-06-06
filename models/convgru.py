import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    """
    Convolutional recurrent cell guiding iterative wind field optimization adjustments.
    
    Input hidden shape: [B, hidden_dim, H/8, W/8]
    Input features x shape: [B, input_dim, H/8, W/8]
    Output hidden shape: [B, hidden_dim, H/8, W/8]
    """
    def __init__(self, hidden_dim=128, input_dim=128):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))

        rh_x = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(rh_x))

        h_next = (1 - z) * h + z * q
        return h_next
