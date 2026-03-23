
import numpy as np
import torch
import torch.nn as nn

class Augmenter(object):

    def __init__(self, cutout_length=4, cutout_prob=0.5, crop_min_history=0.5, crop_prob=0.5, gaussian_std=0.1, dropout_prob=0.1, is_cuda=True):
        self.cutout_length = cutout_length
        self.cutout_prob = cutout_prob
        self.crop_min_history = crop_min_history
        self.crop_prob = crop_prob
        self.gaussian_std = gaussian_std
        self.dropout_prob = dropout_prob
        self.is_cuda = is_cuda
        self.augmentations = [self.history_cutout, self.history_crop, self.gaussian_noise, self.spatial_dropout]

    def __call__(self, sequence, sequence_mask):
        for f in self.augmentations:
            (sequence, sequence_mask) = f(sequence, sequence_mask)
        return (sequence, sequence_mask)

    def history_cutout(self, sequence, sequence_mask):
        (n_seq, n_len, n_channel) = sequence.shape
        cutout_start_index = torch.randint(low=0, high=(n_len - self.cutout_length), size=(n_seq, 1)).expand((- 1), n_len)
        cutout_end_index = (cutout_start_index + self.cutout_length)
        indices_tensor = torch.arange(n_len).repeat(n_seq, 1)
        mask_pre = (indices_tensor < cutout_start_index)
        mask_post = (indices_tensor >= cutout_end_index)
        mask_cutout = (mask_pre + mask_post)
        mask_cutout = mask_cutout.unsqueeze(dim=(- 1)).expand((- 1), (- 1), n_channel).long()
        cutout_selection = (torch.rand(n_seq) < self.cutout_prob).long().reshape((- 1), 1, 1)
        if self.is_cuda:
            cutout_selection = cutout_selection.cuda()
            mask_cutout = mask_cutout.cuda()
            sequence = sequence.cuda()
            sequence_mask = sequence_mask.cuda()
        sequence_cutout = ((sequence * (1 - cutout_selection)) + ((sequence * cutout_selection) * mask_cutout))
        sequence_mask_cutout = ((sequence_mask * (1 - cutout_selection)) + ((sequence_mask * cutout_selection) * mask_cutout))
        return (sequence_cutout, sequence_mask_cutout)

    def history_crop(self, sequence, sequence_mask):
        (n_seq, n_len, n_channel) = sequence.shape
        nonpadded = sequence_mask.sum(dim=(- 1)).cpu()
        first_nonpadded = (self.get_first_nonzero(nonpadded).reshape((- 1), 1) / n_len)
        crop_start_index = torch.rand(size=(n_seq, 1))
        crop_start_index = (((crop_start_index * (1 - first_nonpadded)) * self.crop_min_history) + first_nonpadded)
        crop_start_index = (crop_start_index * n_len).long().expand((- 1), n_len)
        indices_tensor = torch.arange(n_len).repeat(n_seq, 1)
        mask_crop = (indices_tensor >= crop_start_index)
        mask_crop = mask_crop.unsqueeze(dim=(- 1)).expand((- 1), (- 1), n_channel).long()
        crop_selection = (torch.rand(n_seq) < self.crop_prob).long().reshape((- 1), 1, 1)
        if self.is_cuda:
            crop_selection = crop_selection.cuda()
            mask_crop = mask_crop.cuda()
        sequence_crop = ((sequence * (1 - crop_selection)) + ((sequence * crop_selection) * mask_crop))
        sequence_mask_crop = ((sequence_mask * (1 - crop_selection)) + ((sequence_mask * crop_selection) * mask_crop))
        return (sequence_crop, sequence_mask_crop)

    def gaussian_noise(self, sequence, sequence_mask):
        padding_mask = (sequence_mask != 0).long()
        noise = nn.init.trunc_normal_(torch.empty_like(sequence), std=self.gaussian_std, a=((- 2) * self.gaussian_std), b=(2 * self.gaussian_std))
        sequence_noisy = (sequence + (padding_mask * noise))
        return (sequence_noisy, sequence_mask)

    def spatial_dropout(self, sequence, sequence_mask):
        (n_seq, n_len, n_channel) = sequence.shape
        dropout_selection = (torch.rand((n_seq, 1, n_channel)) > self.dropout_prob).long().expand((- 1), n_len, (- 1))
        if self.is_cuda:
            dropout_selection = dropout_selection.cuda()
        sequence_dropout = (sequence * dropout_selection)
        sequence_mask_dropout = (sequence_mask * dropout_selection)
        return (sequence_dropout, sequence_mask_dropout)

    def get_first_nonzero(self, tensor2d):
        nonzero = (tensor2d != 0)
        cumsum = nonzero.cumsum(dim=(- 1))
        nonzero_idx = ((cumsum == 1) & nonzero).max(dim=(- 1)).indices
        return nonzero_idx

def concat_mask(seq, seq_mask, use_mask=False):
    if use_mask:
        seq = torch.cat([seq, seq_mask], dim=2)
    return seq

class Injector(object):

    def __init__(self, seq, portion_len=0.9):
        self.portion_len = portion_len
        self.injected_win = self.inject_anomaly(seq)

    def inject_anomaly(self, window, subsequence_length: int=None, compression_factor: int=None, scale_factor: float=None, trend_factor: float=None, shapelet_factor: bool=False, trend_end: bool=False, start_index: int=None):
        window = window.copy()
        if (subsequence_length is None):
            min_len = int((window.shape[0] * 0.2))
            max_len = int((window.shape[0] * 0.9))
            subsequence_length = np.random.randint(min_len, max_len)
        if (compression_factor is None):
            compression_factor = np.random.randint(2, 5)
        if (scale_factor is None):
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
        if (start_index is None):
            start_index = np.random.randint(0, (len(window) - subsequence_length))
        end_index = min((start_index + subsequence_length), window.shape[0])
        if trend_end:
            end_index = window.shape[0]
        anomalous_subsequence = window[start_index:end_index]
        anomalous_subsequence = np.repeat(anomalous_subsequence, compression_factor, axis=0)
        anomalous_subsequence = anomalous_subsequence[::compression_factor]
        anomalous_subsequence = (anomalous_subsequence * scale_factor)
        if (trend_factor is None):
            trend_factor = np.random.normal(1, 0.5)
        coef = 1
        if (np.random.uniform() < 0.5):
            coef = (- 1)
        anomalous_subsequence = (anomalous_subsequence + (coef * trend_factor))
        if shapelet_factor:
            anomalous_subsequence = (window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape((- 1), 1))
        window[start_index:end_index] = anomalous_subsequence
        return np.squeeze(window)

    def __call__(self, X):
        window = X.copy()
        anomaly_seasonal = np.zeros_like(window)
        anomaly_trend = np.zeros_like(window)
        anomaly_global = np.zeros_like(window)
        anomaly_contextual = np.zeros_like(window)
        anomaly_shapelet = np.zeros_like(window)
        if (window.ndim > 1):
            num_features = window.shape[1]
            min_len = int((window.shape[0] * 0.2))
            max_len = int((window.shape[0] * 0.9))
            subsequence_length = np.random.randint(min_len, max_len)
            start_index = np.random.randint(0, (len(window) - subsequence_length))
            num_dims = np.random.randint(1, int((num_features / 10)))
            for k in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_win = window[:, i].reshape((window.shape[0], 1))
                anomaly_seasonal[:, i] = self.inject_anomaly(temp_win, scale_factor=1, trend_factor=0, subsequence_length=subsequence_length, start_index=start_index)
                anomaly_trend[:, i] = self.inject_anomaly(temp_win, compression_factor=1, scale_factor=1, trend_end=True, subsequence_length=subsequence_length, start_index=start_index)
                anomaly_global[:, i] = self.inject_anomaly(temp_win, subsequence_length=3, compression_factor=1, scale_factor=5, trend_factor=0, start_index=start_index)
                anomaly_contextual[:, i] = self.inject_anomaly(temp_win, subsequence_length=5, compression_factor=1, scale_factor=2, trend_factor=0, start_index=start_index)
                anomaly_shapelet[:, i] = self.inject_anomaly(temp_win, compression_factor=1, scale_factor=1, trend_factor=0, shapelet_factor=True, subsequence_length=subsequence_length, start_index=start_index)
        else:
            temp_win = window.reshape((len(window), 1))
            anomaly_seasonal = self.inject_anomaly(temp_win, scale_factor=1, trend_factor=0)
            anomaly_trend = self.inject_anomaly(temp_win, compression_factor=1, scale_factor=1, trend_end=True)
            anomaly_global = self.inject_anomaly(temp_win, subsequence_length=3, compression_factor=1, scale_factor=5, trend_factor=0)
            anomaly_contextual = self.inject_anomaly(temp_win, subsequence_length=5, compression_factor=1, scale_factor=2, trend_factor=0)
            anomaly_shapelet = self.inject_anomaly(temp_win, compression_factor=1, scale_factor=1, trend_factor=0, shapelet_factor=True)
        anomalies = [anomaly_seasonal, anomaly_trend, anomaly_global, anomaly_contextual, anomaly_shapelet]
        self.anomalous_window = np.random.choice(anomalies)
        return self.anomalous_window
