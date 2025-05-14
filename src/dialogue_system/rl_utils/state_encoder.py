import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import RGATConv, global_mean_pool

from dialogue_system.rl_utils.rl_parameters import STATE_EMBEDDING_SIZE, STATE_HIDDEN_SIZE
from dialogue_system.utils.hp_rdf_dataset import HarryPotterRDF


class EncoderAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        # in_channels is the number_features
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.lin(x)
        x = global_mean_pool(x, None)
        x = F.log_softmax(x, dim=-1)
        return x


class StateEncoder(object):
    def __init__(self, dataset, embedding_size=STATE_EMBEDDING_SIZE, hidden_size=STATE_HIDDEN_SIZE):
        self.dataset = dataset
        self.embedding_size = STATE_EMBEDDING_SIZE
        self.model_attention = EncoderAttention(self.dataset.NUM_FEATURES, hidden_size, embedding_size,
                                                self.dataset.NUM_RELATIONS)

    def encode(self, trig_file):
        with torch.no_grad():  # TODO change this if we do train the encoder
            # RGAT - Conv
            data = self.dataset.process_one_graph(trig_file)

            # Check if the graph is empty,so we return a zero tensor or the right dimensions
            if len(data.edge_type) > 0:
                encoded_state = self.model_attention(data.node_features.float(), data.edge_index, data.edge_type)
            else:
                encoded_state = torch.tensor(np.zeros([1, self.embedding_size]), dtype=torch.float)

        return encoded_state


# def main():
#     dataset = HarryPotterRDF('.')
#     state_encoder = StateEncoder(dataset)
#     encoded_state = state_encoder.encode(dataset.raw_file_names[0])
#     print(f"Encoded the brain!: {encoded_state}")
#
#
# if __name__ == "__main__":
#     main()
