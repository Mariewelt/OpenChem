"""
Extended GraphRNN model that supports edge and node class prediction

"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

from .openchem_model import OpenChemModel


class GraphRNNModel(OpenChemModel):

    def __init__(self, params):
        super(GraphRNNModel, self).__init__(params)
        self.num_node_classes = params["num_node_classes"]
        self.num_edge_classes = params["num_edge_classes"]
        self.max_num_nodes = params["max_num_nodes"]
        self.start_node_label = params["start_node_label"]

        if self.num_edge_classes > 2:
            EdgeEmbedding = params["EdgeEmbedding"]
            edge_embedding_params = params["edge_embedding_params"]
            self.edge_emb = EdgeEmbedding(edge_embedding_params)
        else:
            self.edge_emb = None

        if self.num_node_classes > 2:
            NodeEmbedding = params["NodeEmbedding"]
            node_embedding_params = params["node_embedding_params"]
            self.node_emb = NodeEmbedding(node_embedding_params)

            NodeMLP = params["NodeMLP"]
            node_mlp_params = params["node_mlp_params"]
            self.node_mlp = NodeMLP(node_mlp_params)
        else:
            self.node_emb = None
            self.node_mlp = None

        # TODO: rewrite these two in OpenChem native style
        NodeRNN = params["NodeRNN"]
        node_rnn_params = params["node_rnn_params"]
        self.node_rnn = NodeRNN(**node_rnn_params)

        EdgeRNN = params["EdgeRNN"]
        edge_rnn_params = params["edge_rnn_params"]
        self.edge_rnn = EdgeRNN(**edge_rnn_params)

    def cast_inputs(self, batch):

        batch_input = dict(
            x=torch.tensor(batch["x"], dtype=torch.float),
            y=torch.tensor(batch["y"], dtype=torch.float),
            c_in=torch.tensor(batch["c_in"], dtype=torch.long),
            c_out=torch.tensor(batch["c_out"], dtype=torch.long),
            num_nodes=torch.tensor(batch["num_nodes"], dtype=torch.long),
        )
        batch_target = None

        return batch_input, batch_target

    # TODO: implement required params
    # def get_required_params(self):
    #     return {}

    def forward(self, inp, eval=False):
        if self.training:
            return self.forward_train(inp)
        else:
            return self.forward_test()

    def forward_test(self, batch_size=32):
        device = torch.device("cuda")

        # TODO: where is this function called?
        max_num_node = int(self.max_num_node)
        y_pred_long = torch.zeros(
            batch_size, max_num_node, self.max_prev_node
        ).to(device=device)
        c_pred_long = torch.zeros(
            batch_size, max_num_node
        ).to(device=device)
        x_step = torch.ones(
            batch_size, 1, self.max_prev_node
        ).to(device=device)
        c_step = self.start_node_label * torch.ones(
            batch_size, 1, dtype=torch.long, device=device)

        for i in range(max_num_node):
            if self.edge_emb is not None:
                x_step = self.edge_emb(
                    x_step.to(dtype=torch.long)).view(batch_size, 1, -1)
            if self.node_emb is not None:
                c_step = self.node_emb(c_step)
                x_step = torch.cat([x_step, c_step], dim=2)
            h = self.node_rnn(x_step, return_output_raw=False)
            # output.hidden = h.permute(1,0,2)
            self.edge_rnn.hidden = self.edge_rnn.init_hidden(h.size(0))
            self.edge_rnn.hidden = torch.cat(
                [h.permute(1, 0, 2), self.edge_rnn.hidden[1:]],
                dim=0
            )
            x_step = torch.zeros(batch_size, 1, self.max_prev_node,
                                 device=device)
            output_x_step = torch.ones(batch_size, 1, 1, device=device)
            if self.node_mlp is not None:
                vert_pred = self.node_mlp(h)
                # [test_batch_size x 1 x num_classes]
                vert_pred = F.softmax(vert_pred, dim=2)
                # [test_batch_size x 1]
                c_step = torch.multinomial(
                    vert_pred.view(-1, self.num_node_classes), 1)
                c_pred_long[:, i:i + 1] = c_step
            for j in range(min(self.max_prev_node, i + 1)):
                output_y_pred_step = self.edge_rnn(output_x_step)
                if self.num_edge_classes == 1:
                    output_y_pred_step = F.sigmoid(output_y_pred_step)
                    output_x_step = torch.bernoulli(output_y_pred_step)
                else:
                    output_y_pred_step = F.softmax(output_y_pred_step, dim=-1)
                    output_x_step = torch.multinomial(
                        output_y_pred_step.view(-1, self.num_edge_classes),
                        num_samples=1).view(-1, 1, 1)
                x_step[:, :, j:j + 1] = output_x_step
                # edge rnn keeps the state from current state

            y_pred_long[:, i:i + 1, :] = x_step

        return y_pred_long.to(dtype=torch.long), \
            c_pred_long.to(dtype=torch.long)

    def forward_train(self, inp):
        device = torch.device("cuda")

        x_unsorted = inp["x"]
        y_unsorted = inp["y"]
        c_in_unsorted = inp["c_in"]
        c_out_unsorted = inp["c_out"]
        num_nodes_unsorted = inp["num_nodes"]
        max_num_nodes = num_nodes_unsorted.max().item()
        x_unsorted = x_unsorted[:, 0:max_num_nodes, :]
        y_unsorted = y_unsorted[:, 0:max_num_nodes, :]
        c_in_unsorted = c_in_unsorted[:, 0:max_num_nodes]
        c_out_unsorted = c_out_unsorted[:, 0:max_num_nodes]

        self.node_rnn.hidden = self.node_rnn.init_hidden(
            batch_size=x_unsorted.size(0))

        # sort input samples according to sequence lengths
        num_nodes, sort_index = torch.sort(
            num_nodes_unsorted, 0, descending=True)
        num_nodes = num_nodes.cpu().numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        c_in = torch.index_select(c_in_unsorted, 0, sort_index)
        c_out = torch.index_select(c_out_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function:
        # pack variable -- b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, num_nodes, batch_first=True).data
        c_out = pack_padded_sequence(c_out, num_nodes, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0) - 1, -1, -1)]
        idx = torch.tensor(idx, dtype=torch.long, device=device)
        y_reshape = y_reshape.index_select(0, idx)
        c_out = c_out.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat(
            (torch.ones(y_reshape.size(0), 1, 1, device=device),
             y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(num_nodes))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            # count how many y_len is above i
            count_temp = np.sum(output_y_len_bin[i:])
            # put them in output_y_len; max value should not exceed y.size(2)
            output_y_len.extend(
                [min(i, y.size(2))] * count_temp)

        # pack into variable
        x = x.to(device)
        y = y.to(device)
        c_in = c_in.to(device)
        c_out = c_out.to(device)
        output_x = output_x.to(device)
        output_y = output_y.to(device)

        if self.edge_emb is not None:
            B, T = x.shape[0], x.shape[1]
            x = self.edge_emb(x.to(dtype=torch.long)).view(B, T, -1)

        if self.node_emb is not None:
            # when classes are predicted, have to include previous class
            #  as an input to rnn()
            c_in = self.node_emb(c_in)
            x = torch.cat([x, c_in], dim=2)

        # if using ground truth to train
        # [B x maxL x max_prev_nodes]
        h = self.node_rnn(x, pack=True, input_len=num_nodes,
                          return_output_raw=False)
        # [sum(L) x hidden_dim]
        # get packed hidden vector
        h = pack_padded_sequence(h, num_nodes, batch_first=True).data
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = torch.tensor(idx, dtype=torch.long, device=device)
        h = h.index_select(0, idx)

        self.edge_rnn.hidden = self.edge_rnn.init_hidden(h.size(0))
        self.edge_rnn.hidden = torch.cat(
            [h.unsqueeze(0), self.edge_rnn.hidden[1:]],
            dim=0
        )

        if self.node_mlp is not None:
            node_pred = self.node_mlp(h)

        y_pred = self.edge_rnn(output_x, pack=True, input_len=output_y_len)
        if self.num_edge_classes == 2:
            y_pred = F.sigmoid(y_pred)

        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len,
                                        batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        if self.num_edge_classes == 2:
            loss_edges = F.binary_cross_entropy(y_pred, output_y)
        else:
            loss_edges = F.cross_entropy(y_pred.view(-1, self.num_edge_classes),
                                         output_y.view(-1))
        if self.node_mlp is not None:
            loss_vertices = F.cross_entropy(node_pred, c_out)
            loss_vertices = loss_vertices / self.num_node_classes
            loss = loss_edges + loss_vertices
        else:
            loss = loss_edges

        return loss
