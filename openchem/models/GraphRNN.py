"""
Extended GraphRNN model that supports edge and node class prediction

"""

import numpy as np
import networkx as nx
import torch
from collections import OrderedDict

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

from .openchem_model import OpenChemModel
from openchem.data.graph_utils import decode_adj, SmilesFromGraphs
from openchem.data.utils import sanitize_smiles


class GraphRNNModel(OpenChemModel):

    def __init__(self, params):
        super(GraphRNNModel, self).__init__(params)
        self.num_node_classes = params["num_node_classes"]
        self.num_edge_classes = params["num_edge_classes"]
        self.max_num_nodes = params["max_num_nodes"]
        self.start_node_label = params["start_node_label"]
        self.max_prev_nodes = params["max_prev_nodes"]
        self.label2atom = params["label2atom"]
        self.edge2type = params["edge2type"]

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

        if "from_original" in params.keys():
            paths = params["from_original"]
            self.load_from_original_checkpoint(paths)

    def cast_inputs(self, batch):

        device = torch.device('cpu')

        batch_input = dict(
            x=batch["x"].to(device=device),
            y=batch["y"].to(device=device),
            c_in=batch["c_in"].to(device=device),
            c_out=batch["c_out"].to(device=device),
            num_nodes=batch["num_nodes"].to(device=device),
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

    def forward_test(self, batch_size=1024):
        device = torch.device("cuda")

        # TODO: where is this function called?
        max_num_nodes = int(self.max_num_nodes)
        y_pred_long = torch.zeros(
            batch_size, max_num_nodes, self.max_prev_nodes,
            dtype=torch.long, device=device)
        c_pred_long = torch.zeros(
            batch_size, max_num_nodes, dtype=torch.long, device=device)
        x_step = torch.ones(
            batch_size, 1, self.max_prev_nodes, device=device)
        c_step = self.start_node_label * torch.ones(
            batch_size, 1, dtype=torch.long, device=device)

        self.node_rnn.hidden = self.node_rnn.init_hidden(batch_size, device)
        for i in range(max_num_nodes):
            if self.edge_emb is not None:
                x_step = self.edge_emb(
                    x_step.to(dtype=torch.long)).view(batch_size, 1, -1)
            if self.node_emb is not None:
                c_step = self.node_emb(c_step)
                x_step = torch.cat([x_step, c_step], dim=2)
            h = self.node_rnn(x_step, return_output_raw=False)
            # output.hidden = h.permute(1,0,2)
            self.edge_rnn.hidden = self.edge_rnn.init_hidden(h.size(0), device)
            self.edge_rnn.hidden = torch.cat(
                [h.permute(1, 0, 2), self.edge_rnn.hidden[1:]],
                dim=0
            )
            x_step = torch.zeros(batch_size, 1, self.max_prev_nodes,
                                 device=device)
            output_x_step = torch.ones(batch_size, 1, 1,
                                       device=device, dtype=torch.long)
            if self.node_mlp is not None:
                vert_pred = self.node_mlp(h.view(batch_size, -1))
                # [test_batch_size x num_classes]
                vert_pred = F.softmax(vert_pred, dim=1)
                # [test_batch_size x 1]
                c_step = torch.multinomial(
                    vert_pred.view(-1, self.num_node_classes), 1)
                c_pred_long[:, i:i + 1] = c_step
            for j in range(min(self.max_prev_nodes, i + 1)):
                if self.edge_emb is not None:
                    output_x_step = self.edge_emb(output_x_step).view(
                        batch_size, 1, -1)
                output_y_pred_step = self.edge_rnn(output_x_step)
                if self.num_edge_classes == 2:
                    output_y_pred_step = torch.sigmoid(output_y_pred_step)
                    output_x_step = torch.bernoulli(output_y_pred_step)
                else:
                    output_y_pred_step = F.softmax(output_y_pred_step, dim=-1)
                    output_x_step = torch.multinomial(
                        output_y_pred_step.view(-1, self.num_edge_classes),
                        num_samples=1).view(-1, 1, 1)
                x_step[:, :, j:j + 1] = output_x_step
                # edge rnn keeps the state from current state

            y_pred_long[:, i:i + 1, :] = x_step

        smiles = []
        for i in range(batch_size):
            adj = decode_adj(y_pred_long[i].to('cpu').numpy())
            adj = adj[~np.all(adj == 0, axis=1)]
            adj = adj[:, ~np.all(adj == 0, axis=0)]

            G = nx.from_numpy_matrix(adj)

            num_nodes = len(adj)

            start_c = self.start_node_label
            start_atom = self.label2atom[start_c]
            node_list = [start_atom, ]
            for inode in range(num_nodes - 1):
                c = c_pred_long[i, inode].item()
                atom = self.label2atom[c]
                node_list.append(atom)

            for inode, label in enumerate(node_list):
                G.add_node(inode, label=label)

            G = max(nx.connected_component_subgraphs(G), key=len)
            G = nx.convert_node_labels_to_integers(G)

            node_list = nx.get_node_attributes(G, 'label')
            adj = nx.adj_matrix(G)
            adj = np.array(adj.todense()).astype(int)

            if self.num_edge_classes > 2:
                adj_out = np.zeros(adj.shape)
                for e, t in enumerate(self.edge2type):
                    adj_out[adj == e] = t
            else:
                adj_out = adj

            sstring = SmilesFromGraphs(node_list, adj_out)
            smiles.append(sstring)

        smiles, _ = sanitize_smiles(smiles)
        smiles = [s for s in smiles if len(s)]

        return smiles

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
            batch_size=x_unsorted.size(0), device=device)

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

        if self.num_edge_classes == 2:
            output_x = torch.cat(
                (torch.ones(y_reshape.size(0), 1, 1, device=device),
                 y_reshape[:, 0:-1, 0:1].to(dtype=torch.float)), dim=1)
        else:
            output_x = torch.cat(
                (torch.ones(y_reshape.size(0), 1, 1,
                            device=device, dtype=torch.long),
                 y_reshape[:, 0:-1, 0:1]), dim=1)
            output_x = self.edge_emb(output_x).squeeze()

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

        self.edge_rnn.hidden = self.edge_rnn.init_hidden(h.size(0), device)
        self.edge_rnn.hidden = torch.cat(
            [h.unsqueeze(0), self.edge_rnn.hidden[1:]],
            dim=0
        )

        if self.node_mlp is not None:
            node_pred = self.node_mlp(h)

        y_pred = self.edge_rnn(output_x, pack=True, input_len=output_y_len)
        # if self.num_edge_classes == 2:
        #     y_pred = torch.sigmoid(y_pred)

        # clean
        weights = torch.ones(*output_y.shape, dtype=torch.float, device=device)
        weights = pack_padded_sequence(weights, output_y_len, batch_first=True)
        weights = pad_packed_sequence(weights, batch_first=True)[0]
        # y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        # y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # output_y = pack_padded_sequence(output_y, output_y_len,
        #                                 batch_first=True)
        # output_y = pad_packed_sequence(output_y, batch_first=True)[0]

        # use cross entropy loss
        if self.num_edge_classes == 2:
            # loss_edges = F.binary_cross_entropy(
            #     y_pred, output_y.to(torch.float))
            loss_edges = F.binary_cross_entropy_with_logits(
                y_pred, output_y.to(torch.float), reduction='none')
            n = weights.sum().clamp(min=1.)
            loss_edges = (loss_edges * weights).sum() / n
        else:
            loss_edges = F.cross_entropy(
                y_pred.reshape(-1, self.num_edge_classes),
                output_y.reshape(-1), reduction='none') / self.num_edge_classes
            n = weights.sum().clamp(min=1.)
            loss_edges = (loss_edges * weights.reshape(-1)).sum() / n
        if self.node_mlp is not None:
            loss_vertices = F.cross_entropy(node_pred, c_out,
                                            ignore_index=-1)
            loss_vertices = loss_vertices / self.num_node_classes
            loss = loss_edges + loss_vertices
        else:
            loss = loss_edges

        return loss

    def load_from_original_checkpoint(self, paths):
        prefix_mapping = OrderedDict({
            "lstm.": "node_rnn.",
            "output.": "edge_rnn.",
            "classes_emb.": "node_emb.embedding.",
            "classes.deterministic_output.0": "node_mlp.layers.0",
            "classes.deterministic_output.2": "node_mlp.layers.1"
        })
        params_old = OrderedDict()
        for path in paths:
            params = torch.load(path)
            prefix = None
            for p in ["lstm", "output", "classes", "classes_emb"]:
                if p in path:
                    prefix = p
            assert prefix is not None
            params = {prefix + "." + k: v for k, v in params.items()}
            params_old.update(params)

        params_new = OrderedDict()
        for k, v in params_old.items():
            kn = None
            for po, pn in prefix_mapping.items():
                if k.startswith(po):
                    kn = pn + k[len(po):]
            if kn is None:
                raise AttributeError("Failed to map old key {}".format(k))
            params_new[kn] = v
        if len(params_new) != len(self.state_dict()):
            raise AttributeError("Incomplete mapping of old to new keys")

        self.load_state_dict(params_new)
        print("Successfully  loaded snapshot from original codebase!")
