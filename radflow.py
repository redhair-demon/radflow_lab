import logging
import os
import pickle
from collections import Counter
from typing import Any, Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.initializers import InitializerApplicator
from torch_geometric.nn import GATConv, SAGEConv

from .train import yaml_to_params
from .modules.linear import GehringLinear
from .modules.metrics import get_smape

from .base import BaseModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('radflow')
class RADflow(BaseModel):
    def __init__(self,
                 vocab: Vocabulary,
                 agg_type: str,
                 forecast_length: int = 7,
                 backcast_length: int = 42,
                 test_lengths: List[int] = [7],
                 peek: bool = True,
                 data_path: str = './data/vevo/vevo.hdf5',
                 key2pos_path: str = './data/vevo/vevo.key2pos.pkl',
                 base_model_config: str = None,
                 base_model_weights: str = None,
                 multi_views_path: str = None,
                 test_keys_path: str = None,
                 series_len: int = 63,
                 num_layers: int = 8,
                 hidden_size: int = 128,
                 dropout: float = 0.1,
                 max_neighbours: int = 4,
                 max_agg_neighbours: int = 4,
                 max_eval_neighbours: int = 16,
                 edge_selection_method: str = 'prob',
                 cut_off_edge_prob: float = 0.9,
                 hop_scale: int = 1,
                 n_heads: int = 4,
                 neigh_sample: bool = False,
                 t_total: int = 163840,
                 variant: str = 'separate',
                 end_offset: int = 0,
                 view_missing_p: float = 0,
                 edge_missing_p: float = 0,
                 view_randomize_p: bool = True,
                 forward_fill: bool = True,
                 add_zero_attn: bool = True,
                 add_bias_kv: bool = True,  # having bias term seems important
                 attn_out_proj: bool = True,
                 share_attn_out: bool = False,
                 counterfactual_mode: bool = False,
                 log_space: bool = True,
                 n_hops: int = 1,
                 ignore_test_zeros: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super().__init__(vocab)
        self.views_all = None
        if multi_views_path:
            self.views_all = h5py.File(multi_views_path, 'r')['views']
        input_size = 2 if self.views_all else 1
        self.input_size = input_size
        self.decoder = LSTMDecoder(
            hidden_size, num_layers, dropout, variant, input_size)
        self.mse = nn.MSELoss()
        self.hidden_size = hidden_size
        self.peek = peek
        self.max_neighbours = max_neighbours
        self.max_agg_neighbours = max_agg_neighbours
        self.max_eval_neighbours = max_eval_neighbours
        self.cut_off_edge_prob = cut_off_edge_prob
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.total_length = forecast_length + backcast_length
        self.test_lengths = test_lengths
        self.t_total = t_total
        self.current_t = 0
        self.end_offset = end_offset
        self.neigh_sample = neigh_sample
        self.edge_selection_method = edge_selection_method
        self.attn_out_proj = attn_out_proj
        self.counterfactual_mode = counterfactual_mode
        self.log_space = log_space

        self.evaluate_mode = False
        self.view_missing_p = view_missing_p
        self.edge_missing_p = edge_missing_p
        self.n_hops = n_hops
        self.hop_scale = hop_scale
        self.n_layers = num_layers
        self.ignore_test_zeros = ignore_test_zeros

        self.test_keys = set()
        if test_keys_path and os.path.exists(test_keys_path):
            with open(test_keys_path, 'rb') as f:
                self.test_keys = pickle.load(f)

        # Initialising RandomState is slow!
        self.rs = np.random.RandomState(1234)
        self.edge_rs = np.random.RandomState(63242)
        self.sample_rs = np.random.RandomState(3456)
        self.view_randomize_p = view_randomize_p
        self.forward_fill = forward_fill

        if os.path.exists(data_path):
            self.data = h5py.File(data_path, 'r')
            self.series = self.data['views'][...]
            self.edges = self.data['edges']
            self.masks = self.data['masks']
            self.probs = self.data['probs'] if 'probs' in self.data else None

        if key2pos_path and os.path.exists(key2pos_path):
            with open(key2pos_path, 'rb') as f:
                self.key2pos = pickle.load(f)

        assert agg_type in ['mean', 'none', 'attention', 'sage', 'gat']

        if variant in ['separate', 'h', 'p', 'q']:
            node_size = hidden_size
        elif variant in ['hp']:
            node_size = 2 * hidden_size
        elif variant in ['hpq']:
            node_size = 3 * hidden_size

        self.agg_type = agg_type
        if agg_type in ['mean', 'attention', 'sage', 'gat']:
            self.fc = GehringLinear(node_size, input_size)

        if agg_type == 'attention':
            self.attn = nn.MultiheadAttention(
                node_size, n_heads, dropout=0.1, bias=True,
                add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=None, vdim=None)
            self.proj_alter = GehringLinear(node_size, node_size)
            self.proj_ego = GehringLinear(node_size, node_size)
        elif agg_type == 'sage':
            self.conv = SAGEConv(node_size, node_size)
        elif agg_type == 'gat':
            self.conv = GATConv(node_size, node_size // 4,
                                heads=4, dropout=0.1)

        if n_hops == 2:
            self.hop_rs = np.random.RandomState(4321)
            if agg_type == 'attention':
                self.attn2 = nn.MultiheadAttention(
                    node_size, n_heads, dropout=0.1, bias=True,
                    add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=None, vdim=None)
                if share_attn_out:
                    self.proj_alter2 = self.proj_alter
                    self.proj_ego2 = self.proj_ego
                else:
                    self.proj_alter2 = GehringLinear(node_size, node_size)
                    self.proj_ego2 = GehringLinear(node_size, node_size)
            elif agg_type == 'sage':
                self.conv2 = SAGEConv(node_size, node_size)
            elif agg_type == 'gat':
                self.conv2 = GATConv(node_size, node_size // 4,
                                     heads=4, dropout=0.1)

        self.series_len = series_len
        self.max_start = series_len - self.forecast_length * \
            2 - self.total_length - self.end_offset

        # Shortcut to create new tensors in the same device as the module
        self.register_buffer('_long', torch.LongTensor(1))

        initializer(self)

        self.base_model = None
        if base_model_config and os.path.exists(base_model_weights):
            config = yaml_to_params(base_model_config)
            vocab = Vocabulary.from_params(config.pop('vocabulary'))
            model = Model.from_params(vocab=vocab, params=config.pop('model'))
            if torch.cuda.is_available():
                device = torch.device(f'cuda:0')
            else:
                device = torch.device('cpu')
            best_model_state = torch.load(base_model_weights, device)
            model.load_state_dict(best_model_state)
            model.eval().to(device)
            self.base_model = {'model': model}

    def _forward_full(self, series):
        # series.shape == [batch_size, seq_len]

        X = series
        # X.shape == [batch_size, seq_len]

        X, forecast, f_parts = self.decoder(X)

        return X, forecast, f_parts

    def _get_neighbour_embeds(self, X, keys, start, total_len, eval_step=None, counterfactual=False):
        if self.agg_type == 'none':
            return X

        parents = [-99] * len(keys)
        Xm, masks, neigh_keys = self._construct_neighs(
            X, keys, start, total_len, 1, parents, eval_step, counterfactual)

        scores = None
        if self.agg_type == 'mean':
            Xm = self._aggregate_mean(Xm, masks)
        elif self.agg_type == 'attention':
            Xm, scores = self._aggregate_attn(X, Xm, masks, 1)
        elif self.agg_type in ['gat', 'sage']:
            Xm = self._aggregate_gat(X, Xm, masks, 1)

        X_out = self._pool(X, Xm, 1)
        return X_out, scores, neigh_keys

    def _get_edges_by_probs(self, keys, sorted_keys, key_map, start, total_len, parents):
        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        sorted_probs = self.probs[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        edge_counters = []
        for k, parent in zip(keys, parents):
            key_edges = np.vstack(sorted_edges[key_map[k]])
            key_probs = np.vstack(sorted_probs[key_map[k]])
            cutoff_mask = (key_edges != -1)
            cutoff_mask[:, 1:] = key_probs[:, 1:] <= self.cut_off_edge_prob
            key_edges = key_edges[cutoff_mask]

            # Re-map keys - faster than using loops and Counter
            palette = np.unique(key_edges)
            keys = np.array(range(len(palette)), dtype=np.int32)
            index = np.digitize(key_edges, palette, right=True)
            mapped_key_edges = keys[index]
            # We can even specify a weight matrix (e.g. probs) if needed
            counts = np.bincount(mapped_key_edges)
            # counts = np.bincount(mapped_key_edges, weights=key_flows)

            counter = {palette[i]: count for i, count in enumerate(counts)}
            if parent in counter:
                del counter[parent]
            if k in counter:  # self-loops
                del counter[k]

            edge_counters.append(counter)

        return edge_counters

    def _get_top_edges(self, keys, sorted_keys, key_map, start, total_len, parents):
        sorted_edges = self.edges[sorted_keys, start:start+total_len]
        # sorted_edges.shape == [batch_size, total_len, max_neighs]

        edge_counters = []
        for k, parent in zip(keys, parents):
            key_edges = np.vstack(sorted_edges[key_map[k]])
            key_edges = key_edges[:, :self.max_neighbours]
            mask = key_edges != -1
            key_edges = key_edges[mask]

            counter = Counter()
            counter.update(key_edges)

            if parent in counter:
                del counter[parent]
            if k in counter:  # self-loops
                del counter[k]

            edge_counters.append(counter)

        return edge_counters

    def _construct_neighs(self, X, keys, start, total_len, level, parents=None, eval_step=None, counterfactual=False):
        B, T, E = X.shape

        sorted_keys = sorted(set(keys))
        key_map = {k: i for i, k in enumerate(sorted_keys)}

        if self.edge_selection_method == 'prob':
            edge_counters = self._get_edges_by_probs(
                keys, sorted_keys, key_map, start, total_len, parents)
        elif self.edge_selection_method == 'top':
            edge_counters = self._get_top_edges(
                keys, sorted_keys, key_map, start, total_len, parents)

        # First iteration: grab the top neighbours from each sample
        key_neighs = {}
        max_n_neighs = 1
        neigh_set = set()
        for i, (key, counter) in enumerate(zip(keys, edge_counters)):
            kn = set(counter)
            if not kn:
                continue

            if self.neigh_sample and not self.evaluate_mode:
                pairs = counter.items()
                candidates = np.array([p[0] for p in pairs
                                       if p[0] not in self.test_keys])
                if len(candidates) == 0:
                    continue
                probs = np.array([p[1] for p in pairs
                                  if p[0] not in self.test_keys])
                probs = probs / probs.sum()
                kn = self.sample_rs.choice(
                    candidates,
                    size=min(len(probs[probs > 0]), self.max_agg_neighbours),
                    replace=False,
                    p=probs,
                ).tolist()
            else:
                pairs = Counter(counter).most_common(self.max_eval_neighbours)
                kn = [p[0] for p in pairs]

            key_neighs[key] = list(kn)
            neigh_set |= set(kn)
            max_n_neighs = max(max_n_neighs, len(kn))

        if self.views_all:
            neighs = np.zeros((B, max_n_neighs, total_len, self.input_size),
                              dtype=np.float32)
        else:
            neighs = np.zeros((B, max_n_neighs, total_len), dtype=np.float32)
        n_masks = X.new_zeros(B, max_n_neighs).bool()
        parents = np.full((B, max_n_neighs), -99, dtype=np.uint32)
        neigh_keys = X.new_full((B, max_n_neighs), -1).long()

        neigh_list = sorted(neigh_set)
        end = start + self.total_length
        neigh_map = {k: i for i, k in enumerate(neigh_list)}

        if self.views_all:
            neigh_series = self.views_all[neigh_list, start:end]
        else:
            neigh_series = self.series[neigh_list, start:end]
        neigh_series = neigh_series.astype(np.float32)

        if self.view_missing_p > 0:
            # Don't delete test data during evaluation
            if self.evaluate_mode:
                o_series = neigh_series[:, :self.backcast_length]
            else:
                o_series = neigh_series

            if self.view_randomize_p:
                seeds = [self.epoch, int(self.history['_n_samples']),
                         level, 124241]
                view_p_rs = np.random.RandomState(seeds)
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            seeds = [self.epoch, int(self.history['_n_samples']),
                     level, 52212]
            view_rs = np.random.RandomState(seeds)
            indices = view_rs.choice(np.arange(o_series.size),
                                     replace=False,
                                     size=int(round(o_series.size * prob)))
            o_series[np.unravel_index(indices, o_series.shape)] = -1

            if self.evaluate_mode:
                neigh_series[:, :self.backcast_length] = o_series
            else:
                neigh_series = o_series

        if self.forward_fill:
            if self.views_all:
                NB, NS, NE = neigh_series.shape
                neigh_series = neigh_series.transpose(
                    (0, 2, 1)).reshape(NB, NE * NS)
                mask = neigh_series == -1
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                neigh_series = neigh_series[np.arange(idx.shape[0])[
                    :, None], idx]
                neigh_series = neigh_series.reshape(
                    NB, NE, NS).transpose((0, 2, 1))
            else:
                mask = neigh_series == -1
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                neigh_series = neigh_series[np.arange(idx.shape[0])[
                    :, None], idx]

        neigh_series[neigh_series == -1] = 0

        for i, key in enumerate(keys):
            if key in key_neighs:
                for j, n in enumerate(key_neighs[key]):
                    neighs[i, j] = neigh_series[neigh_map[n]][:total_len]
                    parents[i, j] = key
                    n_masks[i, j] = True
                    neigh_keys[i, j] = n

                    if counterfactual and j == 0:
                        neighs[i, j, -1] = 2 * neighs[i, j, -1]

        out_neigh_keys = neigh_keys.cpu().numpy()
        neighs = torch.from_numpy(neighs).to(X.device)
        if self.views_all:
            neighs = neighs.reshape(
                B * max_n_neighs, total_len, self.input_size)
        else:
            neighs = neighs.reshape(B * max_n_neighs, total_len)
        n_masks = n_masks.reshape(B * max_n_neighs)
        parents = parents.reshape(B * max_n_neighs)
        neigh_keys = neigh_keys.reshape(B * max_n_neighs)
        # neighs.shape == [batch_size * max_n_neighs, seq_len]

        neighs = neighs[n_masks]
        parents = parents[n_masks.cpu().numpy()]
        neigh_keys = neigh_keys[n_masks]
        # neighs.shape == [neigh_batch_size, seq_len]

        if neighs.shape[0] == 0:
            masks = X.new_ones(B, 1, T).bool()
            Xm = X.new_zeros(B, 1, T, E)
            return Xm, masks, out_neigh_keys

        if self.log_space:
            neighs = torch.log1p(neighs)

        if self.base_model is not None and eval_step is not None:
            offset = eval_step + 1
            preds = self.base_model['model'].predict_no_agg(
                neighs[:, :-offset], offset)
            neighs = torch.cat([neighs[:, :-offset], preds], dim=1)

        Xn, _, _ = self._forward_full(neighs)
        # Xn.shape == [neigh_batch_size, seq_len, hidden_size]

        if self.peek:
            Xn = Xn[:, 1:]
        else:
            Xn = Xn[:, :-1]

        if self.n_hops - level > 0:
            if not self.evaluate_mode and self.hop_scale > 1:
                size = int(round(len(Xn) / self.hop_scale))
                idx = self.hop_rs.choice(len(Xn), size=size, replace=False)
            else:
                idx = list(range(len(Xn)))

            sampled_keys = neigh_keys[idx].cpu().tolist()

            Xm_2, masks_2, _ = self._construct_neighs(
                Xn[idx], sampled_keys, start, total_len,
                level + 1, parents[idx], eval_step)
            if self.agg_type == 'mean':
                Xm_2 = self._aggregate_mean(Xm_2, masks_2)
            elif self.agg_type == 'attention':
                Xm_2, _ = self._aggregate_attn(
                    Xn[idx], Xm_2, masks_2, level + 1)
            elif self.agg_type in ['gat', 'sage']:
                Xm_2 = self._aggregate_gat(Xn[idx], Xm_2, masks_2, level + 1)
            Xn[idx] = self._pool(Xn[idx], Xm_2, level + 1).type_as(Xn)

        _, S, E = Xn.shape

        # We plus one to give us option to either peek or not
        Xm = X.new_zeros(B * max_n_neighs, S, E)
        Xm[n_masks] = Xn
        Xm = Xm.reshape(B, max_n_neighs, S, E)

        masks = np.ones((B, max_n_neighs, S), dtype=bool)
        sorted_masks = self.masks[sorted_keys, start:start+total_len]

        for b, key in enumerate(keys):
            if key not in key_neighs:
                continue
            n_mask = np.vstack(sorted_masks[key_map[key]])
            if self.edge_missing_p > 0:
                seeds = [key, self.epoch, int(self.history['_n_samples']),
                         level, 124241]
                edge_rs = np.random.RandomState(seeds)
                D, N = n_mask.shape
                n_mask = n_mask.reshape(-1)
                edge_idx = (~n_mask).nonzero()[0]
                size = int(round(len(edge_idx) * self.edge_missing_p))
                if size > 0:
                    delete_idx = edge_rs.choice(edge_idx,
                                                replace=False,
                                                size=size)
                    n_mask[delete_idx] = True
                n_mask = n_mask.reshape(D, N)
            for i, k in enumerate(key_neighs[key]):
                if not hasattr(self, 'key2pos'):
                    masks[b, i] = 0
                    continue
                mask = n_mask[:, self.key2pos[key][k]]
                if self.peek:
                    mask = mask[1:]
                else:
                    mask = mask[:-1]
                masks[b, i] = mask

        masks = torch.from_numpy(masks).to(X.device)

        return Xm, masks, out_neigh_keys

    def _pool(self, X, Xn, level):
        if self.agg_type == 'mean':
            X_out = X + Xn
        elif self.agg_type == 'attention' and not self.attn_out_proj:
            X_out = X + Xn
            X_out = F.gelu(X_out).type_as(X)
        elif self.agg_type == 'attention' and self.attn_out_proj:
            if level == 1:
                X_out = self.proj_ego(X) + self.proj_alter(Xn)
            elif level == 2:
                X_out = self.proj_ego2(X) + self.proj_alter2(Xn)
            X_out = F.gelu(X_out).type_as(X)
        elif self.agg_type in ['gat', 'sage']:
            X_out = Xn

        return X_out

    def _aggregate_mean(self, Xn, masks):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        # Mask out irrelevant values.
        # Xn = Xn.clone()
        # Xn[masks] = 0

        # Let's just take the average
        Xn = Xn.sum(dim=1)
        # Xn.shape == [batch_size, seq_len, hidden_size]

        n_neighs = (~masks).sum(dim=1).unsqueeze(-1)
        # Avoid division by zero
        n_neighs = n_neighs.clamp(min=1)
        # n_neighs.shape == [batch_size, seq_len, 1]

        Xn = Xn / n_neighs
        # Xn.shape == [batch_size, seq_len, hidden_size]

        return Xn

    def _aggregate_attn(self, X, Xn, masks, level):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        X = X.reshape(1, B * T, E)
        # X.shape == [1, batch_size * seq_len, hidden_size]

        Xn = Xn.transpose(0, 1).reshape(N, B * T, E)
        # Xn.shape == [n_neighs, batch_size * seq_len, hidden_size]

        key_padding_mask = masks.transpose(1, 2).reshape(B * T, N)
        # key_padding_mask.shape == [n_neighs, batch_size  * seq_len]

        return_weights = self.evaluate_mode

        if level == 1:
            X_attn, scores = self.attn(
                X, Xn, Xn, key_padding_mask, return_weights)
        elif level == 2:
            X_attn, scores = self.attn2(
                X, Xn, Xn, key_padding_mask, return_weights)

        # X_attn.shape == [1, batch_size * seq_len, hidden_size]

        X_out = X_attn.reshape(B, T, E)

        if scores is not None:
            scores = scores.reshape(B, T, -1)
            scores = scores.cpu().numpy()

        return X_out, scores

    def _aggregate_gat(self, X, Xn, masks, level):
        # X.shape == [batch_size, seq_len, hidden_size]
        # Xn.shape == [batch_size, n_neighs, seq_len, hidden_size]
        # masks.shape == [batch_size, n_neighs, seq_len]

        B, N, T, E = Xn.shape

        Xn = Xn.reshape(B * N * T, E)
        # Xn.shape == [batch_size * n_neighs * seq_len, hidden_size]

        X_in = X.reshape(B * T, E)
        # X_in.shape == [batch_size * seq_len, hidden_size]

        # The indices 0...(BT - 1) will enumerate the central nodes
        # The indices BT...(BT + BNT - 1) will enumerate the neighbours

        keep_masks = ~masks

        sources = torch.arange(B * T, B * T + B * N * T).long()
        sources = sources.reshape(B, N, T)
        sources = sources[keep_masks]

        central_nodes = torch.arange(B * T).long()
        targets = central_nodes.reshape(B, 1, T)
        targets = targets.expand_as(masks)
        targets = targets[keep_masks]

        # Add self-loops to central nodes
        sources = torch.cat([central_nodes, sources])
        targets = torch.cat([central_nodes, targets])
        edges = torch.stack([sources, targets]).to(X.device)

        nodes = torch.cat([X_in, Xn], dim=0)
        # nodes.shape == [BT + BNT, hidden_size]

        if level == 1:
            nodes = self.conv(nodes, edges)
        elif level == 2:
            nodes = self.conv2(nodes, edges)
        else:
            raise NotImplementedError()
        # nodes.shape == [BT + BNT, hidden_size]

        nodes = nodes[:B * T]
        # nodes.shape == [BT, hidden_size]

        # nodes = F.elu(nodes)
        # nodes.shape == [BT, hidden_size]

        X_agg = nodes.reshape(B, T, E)
        # X_agg.shape == [batch_size, seq_len, hidden_size]

        X_agg = F.gelu(X_agg).type_as(X_agg)

        return X_agg

    def forward(self, keys, splits) -> Dict[str, Any]:
        # Enable anomaly detection to find the operation that failed to compute
        # its gradient.
        # torch.autograd.set_detect_anomaly(True)

        # self._initialize_series()

        # Occasionally we get duplicate keys due random sampling
        keys = sorted(set(keys))
        split = splits[0]
        B = len(keys)
        p = next(self.parameters())
        # keys.shape == [batch_size]

        self.history['_n_batches'] += 1
        self.history['_n_samples'] += B

        out_dict = {
            'loss': None,
            'sample_size': p.new_tensor(B),
        }

        if split == 'train':
            if self.max_start == 0:
                start = 0
            else:
                start = self.rs.randint(0, self.max_start)
        elif split == 'valid':
            start = self.max_start + self.forecast_length
        elif split == 'test':
            start = self.max_start + self.forecast_length * 2

        # Find all series of given keys
        end = start + self.total_length

        if self.views_all:
            series = self.views_all[keys, start:end].astype(np.float32)
        else:
            series = self.series[keys, start:end].astype(np.float32)

        if self.view_missing_p > 0:
            # Don't delete test data during evaluation
            if self.evaluate_mode:
                o_series = series[:, :self.backcast_length]
            else:
                o_series = series

            if self.view_randomize_p:
                seeds = [self.epoch, int(self.history['_n_samples']), 6235]
                view_p_rs = np.random.RandomState(seeds)
                prob = view_p_rs.uniform(0, self.view_missing_p)
            else:
                prob = self.view_missing_p
            seeds = [self.epoch, int(self.history['_n_samples']), 12421]
            view_rs = np.random.RandomState(seeds)
            indices = view_rs.choice(np.arange(o_series.size),
                                     replace=False,
                                     size=int(round(o_series.size * prob)))
            o_series[np.unravel_index(indices, o_series.shape)] = -1

            if self.evaluate_mode:
                series[:, :self.backcast_length] = o_series
            else:
                series = o_series

        non_missing_idx = torch.from_numpy(series[:, 1:] != -1).to(p.device)

        if self.forward_fill:
            if self.views_all:
                B, S, E = series.shape
                series = series.transpose((0, 2, 1)).reshape(B, E * S)
                mask = series == -1
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                series = series[np.arange(idx.shape[0])[:, None], idx]
                series = series.reshape(B, E, S).transpose((0, 2, 1))
            else:
                mask = series == -1
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                series = series[np.arange(idx.shape[0])[:, None], idx]

        series[series == -1] = 0
        raw_series = torch.from_numpy(series).to(p.device)
        # raw_series.shape == [batch_size, seq_len]

        # non_missing_idx = torch.stack(non_missing_list, dim=0)[:, 1:]

        if self.log_space:
            log_raw_series = torch.log1p(raw_series)
            series = torch.log1p(raw_series)
        else:
            log_raw_series = raw_series
            series = raw_series

        X_full, preds_full, _ = self._forward_full(series)
        preds = preds_full[:, :-1]
        # X.shape == [batch_size, seq_len, hidden_size]

        if self.agg_type != 'none':
            X = X_full[:, :-1]
            X_agg, _, _ = self._get_neighbour_embeds(
                X, keys, start, self.total_length)
            # X_agg.shape == [batch_size, seq_len, out_hidden_size]

            X_agg = self.fc(X_agg)
            # X_agg.shape == [batch_size, seq_len, 1]

            preds = preds + X_agg.squeeze(-1)
            # preds.shape == [batch_size, seq_len]

        if self.log_space:
            preds = torch.exp(preds)
        targets = raw_series[:, 1:]

        preds = torch.masked_select(preds, non_missing_idx)
        targets = torch.masked_select(targets, non_missing_idx)

        numerator = torch.abs(targets - preds)
        denominator = torch.abs(targets) + torch.abs(preds)
        loss = numerator / denominator
        loss[torch.isnan(loss)] = 0
        loss = loss.mean()
        out_dict['loss'] = loss

        # During evaluation, we compute one time step at a time
        if split in ['valid', 'test']:
            s = self.backcast_length
            e = s + self.forecast_length
            targets = raw_series[:, s:e]
            # targets.shape == [batch_size, forecast_len]

            preds = targets.new_zeros(*targets.shape)
            if self.counterfactual_mode:
                preds_2 = targets.new_zeros(*targets.shape)
                all_scores_2 = [[] for _ in keys]

            series = log_raw_series[:, :-self.forecast_length]
            current_views = series[:, -1]
            all_f_parts = [[[] for _ in range(self.n_layers + 1)]
                           for _ in keys]
            all_scores = [[] for _ in keys]
            for i in range(self.forecast_length):
                X, pred_base, f_parts = self._forward_full(series)
                pred_base = pred_base[:, -1]
                for b in range(len(keys)):
                    for l, f_part in enumerate(f_parts):
                        all_f_parts[b][l].append(f_part[b])
                if self.agg_type != 'none':
                    seq_len = self.total_length - self.forecast_length + i + 1
                    X_agg, scores, neigh_keys = self._get_neighbour_embeds(
                        X, keys, start, seq_len, i)
                    X_agg = self.fc(X_agg)
                    X_agg = X_agg.squeeze(-1)[:, -1]
                    if scores is not None:
                        scores = scores[:, -1].tolist()
                    pred = pred_base + X_agg
                    # delta.shape == [batch_size]

                    if self.counterfactual_mode:
                        X_agg_2, scores_2, _ = self._get_neighbour_embeds(
                            X, keys, start, seq_len, i, True)
                        X_agg_2 = self.fc(X_agg_2)
                        X_agg_2 = X_agg_2.squeeze(-1)[:, -1]
                        if scores_2 is not None:
                            scores_2 = scores_2[:, -1].tolist()
                        pred_2 = pred_base + X_agg_2

                    for b, f in enumerate(X_agg.cpu().tolist()):
                        all_f_parts[b][-1].append(f)
                        if scores is not None:
                            all_scores[b].append(scores[b])
                        if self.counterfactual_mode and scores_2 is not None:
                            all_scores_2[b].append(scores_2[b])

                else:
                    neigh_keys = np.array([])
                    pred = pred_base

                current_views = pred
                preds[:, i] = current_views
                if self.counterfactual_mode:
                    preds_2[:, i] = pred_2
                current_views = current_views.unsqueeze(1)
                series = torch.cat([series, current_views], dim=1)

            if self.log_space:
                preds = torch.exp(preds)
                if self.counterfactual_mode:
                    preds_2 = torch.exp(preds_2)

            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

            if self.ignore_test_zeros:
                nz = targets != 0
                targets = targets[nz]
                preds = preds[nz]

            smapes, daily_errors = get_smape(targets, preds)
            # if self.views_all:
            #     n_cats = smapes.shape[-1]
            #     for i in range(n_cats):
            #         for k in self.test_lengths:
            #             self.step_history[f'smape_{i}_{k}'] += np.sum(
            #                 smapes[:, :k, i])

            rmse = (targets - preds)**2
            mae = np.abs(targets - preds)

            out_dict['smapes'] = smapes.tolist()
            out_dict['daily_errors'] = daily_errors.tolist()
            out_dict['keys'] = keys
            out_dict['preds'] = preds.tolist()
            out_dict['f_parts'] = all_f_parts
            out_dict['neigh_keys'] = neigh_keys.tolist()
            out_dict['all_scores'] = all_scores
            if self.counterfactual_mode:
                out_dict['preds_2'] = preds_2.cpu().numpy().tolist()
                out_dict['all_scores_2'] = all_scores_2
            if self.views_all:
                self.history['_n_steps'] += smapes.shape[0] * \
                    smapes.shape[1] * smapes.shape[2]
            elif len(smapes.shape) == 2:
                self.history['_n_steps'] += smapes.shape[0] * smapes.shape[1]
            else:
                self.history['_n_steps'] += smapes.shape[0]

            k = self.test_lengths[-1]
            self.step_history[f'smape_{k}'] += np.sum(smapes)
            self.squared_step_history[f'_rmse_{k}'] += np.sum(rmse)
            self.step_history[f'_mae_{k}'] += np.sum(mae)
        else:
            self.current_t += 1

        return out_dict

    def get_pred_neigh_embeds(self, X, Xm, masks):
        # X.shape == [batch_size, 1, hidden_size]
        # Xm.shape == [batch_size, n_neighs, 1, hidden_size]
        # masks.shape == [batch_size, n_neighs, 1]

        Xm, scores = self._aggregate_attn(X, Xm, masks, 1)
        X_out = self._pool(X, Xm, 1)
        # X_out.shape == [batch_size, 1, hidden_size]

        return X_out

    def predict(self, backcast, neighs, forecast_len=28):
        assert self.n_hops == 1
        assert backcast.shape[0] == 1
        p = next(self.parameters())

        backcast = p.new_tensor(backcast)
        backcast = torch.log1p(backcast)
        B, _ = backcast.shape
        # X.shape == [batch_size, backcast_len]

        neighs = p.new_tensor(neighs)
        neighs = torch.log1p(neighs)
        B, N, T = neighs.shape
        # neighs.shape == [batch_size, n_neighs, total_len]

        masks = np.zeros((B, N, 1), dtype=bool)
        masks = p.new_tensor(masks)
        preds = p.new_zeros(B, forecast_len)

        X_neighs, _, _ = self._forward_full(neighs.reshape(B * N, T))
        X_neighs = X_neighs.reshape(B, N, T, -1)
        X_neighs = X_neighs[:, :, -forecast_len:]
        # neighs.shape == [batch_size, n_neighs, forecast_len, hidden_size]

        for i in range(forecast_len):
            X, pred, _ = self._forward_full(backcast)
            pred = pred[:, -1:]
            X = X[:, -1:]

            if self.agg_type != 'none':
                X_agg = self.get_pred_neigh_embeds(X, X_neighs[:, :, i], masks)
                X_agg = self.fc(X_agg)
                X_agg = X_agg.squeeze(-1)
                pred = pred + X_agg

            preds[:, i] = pred[:, 0]
            backcast = torch.cat([backcast, pred], dim=1)

        if self.log_space:
            preds = torch.exp(preds)
        # preds.shape == [batch_size, forecast_len]

        return preds.cpu().numpy()

    def predict_no_agg(self, series, n_steps):
        if len(series.shape) == 3:
            preds = series.new_zeros(series.shape[0], n_steps, series.shape[2])
        else:
            preds = series.new_zeros(series.shape[0], n_steps)

        for i in range(n_steps):
            X, pred, f_parts = self._forward_full(series)
            pred = pred[:, -1]
            current_views = pred
            preds[:, i] = current_views
            current_views = current_views.unsqueeze(1)
            series = torch.cat([series, current_views], dim=1)

        return preds


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, variant, input_size):
        super().__init__()
        assert variant in ['none', 'h', 'p', 'q',
                           'hp', 'hpq', 'separate']
        self.variant = variant
        self.input_size = input_size
        self.in_proj = GehringLinear(input_size, hidden_size)
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(LSTMLayer(hidden_size, dropout, variant))

        self.out_f = GehringLinear(hidden_size, input_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        # X.shape == [batch_size, seq_len, 1]

        X = self.in_proj(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        forecast = X.new_zeros(*X.shape)
        if self.variant == 'hp':
            hidden = X.new_zeros(X.shape[0], X.shape[1], 2 * X.shape[2])
        elif self.variant == 'hpq':
            hidden = X.new_zeros(X.shape[0], X.shape[1], 3 * X.shape[2])
        elif self.variant in ['separate', 'h', 'p', 'q']:
            hidden = X.new_zeros(*X.shape)
        else:
            hidden = None
        f_parts = []

        for layer in self.layers:
            h, b, f = layer(X)
            X = X - b
            if self.variant == 'hp':
                hidden = hidden + torch.cat([h, b], dim=-1)
            elif self.variant == 'hpq':
                hidden = hidden + torch.cat([h, b, f], dim=-1)
            elif self.variant == 'h':
                hidden = hidden + h
            elif self.variant == 'p':
                hidden = hidden + b
            elif self.variant == 'q':
                hidden = hidden + f
            elif self.variant == 'separate':
                hidden = hidden + h

            forecast = forecast + f

            if not self.training:
                f_part = self.out_f(f[:, -1]).squeeze(-1)
                f_parts.append(f_part.cpu().tolist())

        if self.variant != 'none':
            hidden = hidden / len(self.layers)

        # h = torch.cat(h_list, dim=-1)
        # h.shape == [batch_size, seq_len, n_layers * hidden_size]

        # h = self.out_proj(h)
        # h.shape == [batch_size, seq_len, hidden_size]

        f = self.out_f(forecast)

        if self.input_size == 1:
            f = f.squeeze(-1)

        return hidden, f, f_parts


class LSTMLayer(nn.Module):
    def __init__(self, hidden_size, dropout, variant):
        super().__init__()
        assert variant in ['none', 'h', 'p', 'q',
                           'hp', 'hpq', 'combined', 'separate']
        self.variant = variant

        self.layer = nn.LSTM(hidden_size, hidden_size, 1,
                             batch_first=True)
        self.drop = nn.Dropout(dropout)

        self.proj_f = GehringLinear(hidden_size, hidden_size)
        self.proj_b = GehringLinear(hidden_size, hidden_size)
        self.out_f = GehringLinear(hidden_size, hidden_size)
        self.out_b = GehringLinear(hidden_size, hidden_size)

        if variant == 'separate':
            self.proj_h = GehringLinear(hidden_size, hidden_size)
            self.out_h = GehringLinear(hidden_size, hidden_size)

    def forward(self, X):
        # X.shape == [batch_size, seq_len]
        # yn.shape == [batch_size, n_neighs, seq_len]

        # It's recommended to apply dropout on the input to the LSTM cell
        # See https://ieeexplore.ieee.org/document/7333848
        X = self.drop(X)

        X, _ = self.layer(X)
        # X.shape == [batch_size, seq_len, hidden_size]

        b = self.out_b(F.gelu(self.proj_b(X)))
        f = self.out_f(F.gelu(self.proj_f(X)))
        # b.shape == f.shape == [batch_size, seq_len, hidden_size]

        if self.variant == 'separate':
            X = self.out_h(F.gelu(self.proj_h(X)))

        return X, b, f
