"""
Extended GraphRNN model that supports edge and node class prediction

"""

import numpy as np
import torch
from collections import OrderedDict

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

from .openchem_model import OpenChemModel
from openchem.data.graph_utils import SmilesFromGraphs, decode_adj_new
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
        self.restrict_min_atoms = params["restrict_min_atoms"]
        self.restrict_max_atoms = params["restrict_max_atoms"]
        self.use_external_criterion = params["use_external_criterion"]

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

        device = torch.device('cpu')

        batch_input = dict(
            x=batch["x"].to(device=device),
            y=batch["y"].to(device=device),
            c_in=batch["c_in"].to(device=device),
            c_out=batch["c_out"].to(device=device),
            num_nodes=batch["num_nodes"].to(device=device),
        )
        batch_target = [None]

        return batch_input, batch_target

    # TODO: implement required params
    # def get_required_params(self):
    #     return {}

    def forward(self, inp, eval=False):
        if self.use_external_criterion:
            assert self.training, "Must be in training mode for external criterion"
            # generate batch
            # self.eval()
            with torch.no_grad():
                inp, smiles = self.forward_test()
            # self.train()
            logp, sizes, smiles = self.forward_train(inp, smiles)
            return logp, sizes, smiles
        elif self.training:
            return self.forward_train(inp)
        else:
            batch, smiles = self.forward_test()
            return smiles

    def forward_test(self, batch_size=1024):
        device = torch.device("cuda")

        # TODO: handle float type for x_step in case of no node embedding
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
        # start generation from second vertex
        prev_num = torch.ones(batch_size, device=device, dtype=torch.float)

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
                                 device=device, dtype=torch.long)
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
                valid_edge = (j < prev_num).to(dtype=x_step.dtype)
                x_step[:, :, j:j + 1] = output_x_step * valid_edge.view(-1, 1, 1)
                # edge rnn keeps the state from current state

            y_pred_long[:, i:i + 1, :] = x_step

            # all zeros in x_step mean that the state is terminal
            # zero out corresponding hidden states
            not_terminal = (x_step.sum(dim=-1) > 0).to(
                dtype=torch.float).view(-1)
            self.node_rnn.hidden *= not_terminal.view(1, -1, 1)

            prev_num = (prev_num * not_terminal) + 1

            terminal = (1 - not_terminal).to(dtype=torch.uint8)
            n_terminal = terminal.sum().item()
            x_step[terminal] = torch.ones(n_terminal, 1, self.max_prev_nodes,
                                          device=device, dtype=torch.long)
            c_step[terminal] = self.start_node_label * torch.ones(
                n_terminal, 1, dtype=torch.long, device=device)

        y_pred_long = y_pred_long.to('cpu').numpy()
        c_pred_long = c_pred_long.to('cpu').numpy()
        smiles = []
        adj_all, classes_all, len_all = [], [], []
        for i in range(batch_size):
            adj_encoded_full = y_pred_long[i]
            # these are vertices with no connections to previous
            #  i.e. last vertex for current sample, and first vertex for
            #  next sample
            anchors = np.where(np.all(adj_encoded_full == 0, axis=1))[0]
            cur = 0
            for nxt in anchors:
                # slice adjacency matrix into connected components
                adj_encoded = adj_encoded_full[cur:nxt, :]
                start_c = self.start_node_label
                atoms = [self.label2atom[start_c], ]
                atoms += [self.label2atom[c] for c in c_pred_long[i, cur:nxt]]

                adj_all.append(torch.from_numpy(adj_encoded))
                classes_all.append(torch.from_numpy(c_pred_long[i, cur:nxt]))
                len_all.append(nxt - cur + 1)

                cur = nxt + 1

                adj = decode_adj_new(adj_encoded)

                remap = self.edge2type if self.num_edge_classes > 2 else None
                sstring = SmilesFromGraphs(atoms, adj, remap=remap)
                smiles.append(sstring)

        # TODO: think how to avoid double sanitization
        smiles, idx = sanitize_smiles(
            smiles, min_atoms=self.restrict_min_atoms,
            max_atoms=self.restrict_max_atoms,
            allowed_tokens=r'#()+-/123456789=@BCFHINOPS[\]cilnors ',
            logging="info"
        )

        smiles = [s for i, s in enumerate(smiles) if i in idx]

        smiles, idx2 = sanitize_smiles(smiles,
                                       min_atoms=self.restrict_min_atoms,
                                       max_atoms=self.restrict_max_atoms,
                                       logging="info")
        idx = [idx[i] for i in idx2]
        smiles = [s for i, s in enumerate(smiles) if i in idx2]

        if self.use_external_criterion:
            adj_all = [s for i, s in enumerate(adj_all) if i in idx]
            classes_all = [s for i, s in enumerate(classes_all) if i in idx]
            len_all = [s for i, s in enumerate(len_all) if i in idx]

            max_len = max(len_all)
            x = torch.zeros(len(smiles), max_len, self.max_prev_nodes,
                            dtype=torch.long)
            x[:, 0, :] = 1.
            y = torch.zeros(len(smiles), max_len, self.max_prev_nodes,
                            dtype=torch.long)
            c_in = torch.zeros(len(smiles), self.max_num_nodes, dtype=torch.long)
            c_in[:, 0] = self.start_node_label
            c_out = -1 * torch.ones(
                len(smiles), self.max_num_nodes, dtype=torch.long)
            for i, (adj, classes, num_nodes) in enumerate(
                    zip(adj_all, classes_all, len_all)):
                y[i, :num_nodes - 1, :] = adj
                x[i, 1:num_nodes, :] = adj
                c_in[i, 1:num_nodes] = classes
                c_out[i, :num_nodes - 1] = classes
            num_nodes = torch.tensor(len_all)
            batch = {"x": x.to(device),
                     "y": y.to(device),
                     "c_in": c_in.to(device),
                     "c_out": c_out.to(device),
                     "num_nodes": num_nodes.to(device)}
        else:
            batch = None

        return batch, smiles

    def forward_train(self, inp, smiles=None):
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
        if smiles is not None:
            smiles = [smiles[i] for i in sort_index.to('cpu').numpy()]

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

        if self.use_external_criterion:
            node_pred = F.log_softmax(node_pred, dim=-1)
            # valid edges are those that do not have all zeros in a row
            valid = torch.ones_like(y_pred)
            valid = pack_padded_sequence(valid, output_y_len, batch_first=True)
            valid = pad_packed_sequence(valid, batch_first=True)[0]
            y_pred = F.log_softmax(y_pred, dim=-1) * valid

            valid = (c_out >= 0).to(node_pred.dtype)
            node_pred = node_pred.gather(
                1, c_out.view(-1, 1).clamp(min=0)).view(-1) * valid
            y_pred = y_pred.gather(2, output_y).squeeze(2)

            sum_y_pred = y_pred.sum(dim=1)

            idx = [i for i in range(sum_y_pred.size(0) - 1, -1, -1)]
            idx = torch.tensor(idx, dtype=torch.long, device=device)
            sum_y_pred = sum_y_pred.index_select(0, idx)

            return node_pred + sum_y_pred, num_nodes, smiles
        else:
            # internal criterion
            weights = torch.ones(*output_y.shape, dtype=torch.float, device=device)
            weights = pack_padded_sequence(weights, output_y_len, batch_first=True)
            weights = pad_packed_sequence(weights, batch_first=True)[0]

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

    def load_model(self, path):
        super(GraphRNNModel, self).load_model(path)
        # TODO: load from original checkpoint if previous line fails
        # self.load_from_original_checkpoint(path)

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
