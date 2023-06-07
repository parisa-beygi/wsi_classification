import torch
from torch import nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F



class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
            dropout, return_embeds=False):
        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        self.convs = nn.ModuleList([pyg_nn.GCNConv(in_channels=input_dim, out_channels=hidden_dim)])
        for i in range(num_layers - 2):
            self.convs.append(pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs.append(pyg_nn.GCNConv(in_channels=hidden_dim, out_channels=output_dim))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(num_layers - 1)])

        self.softmax = nn.LogSoftmax(dim=1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        num_layers = len(self.convs)
        # print (self.bns)
        # print (f'here {self.convs}')
        for i in range(num_layers):
            x = self.convs[i](x, edge_index)
            # print (f'gnn i = {i}')
            # print (f'gnn x.shape = {x.shape}')
            if i < num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, self.training)
            else:
                out = self.softmax(x) if not self.return_embeds else x

		#########################################

        return out


class GCN_Graph(nn.Module):
    def __init__(self, input_dim = 1024, hidden_dim = 512, output_dim = 2, num_layers = 3, dropout = 0.25, pooling = 'mean', linear_hidden_sizes = []):
        super(GCN_Graph, self).__init__()

        # Convert the input_dim node features to hidden_dim node features
        # self.node_encoder = nn.Linear(input_dim, hidden_dim)


        # Node embedding model
        # input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
                            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pooling = pooling
        if pooling == "mean":
            self.pool = pyg_nn.global_mean_pool
        elif pooling == "attention":
            self.pool = pyg_nn.GlobalAttention(nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh()))

        # Output layer
        mlp = []
        first_dim = hidden_dim
        for i in linear_hidden_sizes:
            mlp.append(nn.Linear(first_dim, i))
            first_dim = i
            mlp.append(nn.ReLU())
        
        mlp.append(nn.Linear(first_dim, output_dim))
        
        self.linear = nn.Sequential(*mlp)
        # self.linear = nn.Linear(hidden_dim, output_dim)




    def reset_parameters(self):
        # self.node_encoder.reset_parameters()
        # self.node_encoder.reset_parameters()
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()
        if self.pooling == "attention":
            self.pool.reset_parameters()

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_node = self.gnn_node.to(device)
        self.linear = self.linear.to(device)
        if self.pooling == "attention":
            self.pool = self.pool.to(device)


    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        # x = self.node_encoder(x)
        # x = F.relu(x)
        # x = F.dropout(x, 0.25)
        x = self.gnn_node(x, edge_index)
        # print (f'here gnn_node.shape = {x.shape}')

        x = self.pool(x, batch)
        logits = self.linear(x)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat



class GAT_H(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, heads):
        super(GAT_H, self).__init__()

        self.num_layers = num_layers

        # # Convert the input_dim node features to hidden_dim node features
        # self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.convs = nn.ModuleList([pyg_nn.GATConv(input_dim, hidden_dim, heads=heads)])
        for i in range(num_layers - 1):
            self.convs.append(pyg_nn.GATConv(heads*hidden_dim, hidden_dim, heads=heads))

        self.bns = nn.ModuleList([nn.BatchNorm1d(heads*hidden_dim) for i in range(num_layers - 1)])

        # Probability of an element getting zeroed
        self.dropout = dropout

        self.pool = pyg_nn.global_mean_pool


        self.fc1 = nn.Linear(heads*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)



    def reset_parameters(self):
        # self.node_encoder.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convs = self.convs.to(device)
        self.bns = self.bns.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x = self.node_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, self.dropout, self.training)


        x = self.pool(x, batch)

        out = self.fc1(x)
        logits = self.fc2(out)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat


