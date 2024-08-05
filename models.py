import torch.nn
import misc_functions as misc

class Energy2D(torch.nn.Module):
    def __init__(self, dim_l_embed):
        super(Energy2D, self).__init__()
        
        self.dim_l_embed = dim_l_embed

        internal_dim = 128 + 64

        self.lin_res = torch.nn.Linear(in_features=2*2*dim_l_embed, out_features=internal_dim)

        self.lin_1 = torch.nn.Linear(in_features=2*2*dim_l_embed, out_features=internal_dim)
        self.lin_2 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_3 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_4 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_5 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_6 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_7 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_8 = torch.nn.Linear(in_features=internal_dim, out_features=3)

        self.sigmoid = torch.nn.Sigmoid()

        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        if x.size()[-1] == 2:
            x = misc.positional_encoding(x, self.dim_l_embed)

        x_res = self.lin_res(x)

        x = self.relu(self.lin_1(x))
        x = self.relu(self.lin_2(x))
        x = self.relu(self.lin_3(x))
        x = self.relu(self.lin_4(x))
        x = self.relu(self.lin_5(x+x_res))
        x = self.relu(self.lin_6(x))
        x = self.relu(self.lin_7(x))
        x = self.sigmoid(self.lin_8(x))
        return x

class Deform2d(torch.nn.Module):
    def __init__(self, dim_l_embed):
        super(Deform2d, self).__init__()
        
        self.dim_l_embed = dim_l_embed

        internal_dim = 64

        self.lin_res = torch.nn.Linear(in_features=2*2*dim_l_embed, out_features=internal_dim)

        self.lin_1 = torch.nn.Linear(in_features=2*2*dim_l_embed, out_features=internal_dim)
        self.lin_2 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_3 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_4 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_5 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_6 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_7 = torch.nn.Linear(in_features=internal_dim, out_features=internal_dim)
        self.lin_8 = torch.nn.Linear(in_features=internal_dim, out_features=1)

        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, x):
        if x.size()[-1] == 2:
            x = misc.positional_encoding(x, self.dim_l_embed)
            
        x_res = self.lin_res(x)

        x = self.relu(self.lin_1(x))
        x = self.relu(self.lin_2(x))
        x = self.relu(self.lin_3(x))
        x = self.relu(self.lin_4(x))
        x = self.relu(self.lin_5(x+x_res))
        x = self.relu(self.lin_6(x))
        x = self.relu(self.lin_7(x))
        x = self.lin_8(x)
        return x