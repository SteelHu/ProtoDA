import hashlib
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.augmentations import Injector


def get_dataset(args, domain_type, split_type):
    window_size = getattr(args, 'window_size', 100)
    window_stride = getattr(args, 'window_stride', 1)
    seed = getattr(args, 'seed', 42)
    train_ratio = float(getattr(args, 'train_ratio', 0.8))
    ds = args.dataset
    if (ds == 'alfa'):
        if domain_type == 'source':
            return ALFADataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True, window_size=window_size, window_stride=window_stride, seed=seed, train_ratio=train_ratio, domain_source=True)
        return ALFADataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True, window_size=window_size, window_stride=window_stride, seed=seed, train_ratio=train_ratio, domain_source=False)
    if (ds == 'rflymad'):
        if domain_type == 'source':
            return RflyMADDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True, window_size=window_size, window_stride=window_stride, seed=seed, train_ratio=train_ratio, domain_source=True)
        return RflyMADDataset_trg(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True, window_size=window_size, window_stride=window_stride, seed=seed, train_ratio=train_ratio, domain_source=False)
    raise ValueError('dataset must be alfa or rflymad, got %r' % (ds,))


def split_rng_seed(base_seed, subject_id, domain_source):
    payload = ('%s|%s|%s' % (base_seed, subject_id, ('src' if domain_source else 'trg'))).encode('utf-8', errors='replace')
    return int(int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little') % (2 ** 31))


def window_train_holdout_split(n, labels, random_state, train_ratio=0.8):
    indices = np.arange(n, dtype=int)
    if n <= 0:
        return indices.copy(), np.array([], dtype=int)
    if n == 1:
        return indices.copy(), np.array([], dtype=int)
    n_train = int(train_ratio * n)
    n_train = max(1, min(n - 1, n_train))
    try:
        (train_idx, hold_idx) = train_test_split(indices, train_size=n_train, stratify=labels, random_state=(random_state % (2 ** 31)), shuffle=True)
    except ValueError:
        rng = np.random.default_rng((random_state % (2 ** 31)))
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:n_train])
        hold_idx = np.sort(perm[n_train:])
    return (np.asarray(train_idx, dtype=int), np.asarray(hold_idx, dtype=int))


def finalize_split_windows(windows, wlabels, split_type, domain_source, seed, train_ratio, subject_id, verbose=False, log_prefix=''):
    if (len(windows) == 0):
        n_ch = 1
        return (windows, wlabels, np.zeros(n_ch, dtype=np.float64), np.ones(n_ch, dtype=np.float64))
    if ((not domain_source) and (split_type == 'val')):
        raise ValueError('Target domain has no val split; use train or test.')
    rs = split_rng_seed(seed, subject_id, domain_source)
    (train_idx, hold_idx) = window_train_holdout_split(len(windows), wlabels, rs, train_ratio=train_ratio)
    if (split_type == 'train'):
        idx = train_idx
    elif (split_type in ('val', 'test')):
        idx = (hold_idx if (len(hold_idx) > 0) else train_idx)
        if ((len(hold_idx) == 0) and verbose):
            print(('%s: holdout empty; using train indices.' % log_prefix))
    else:
        raise ValueError("split_type must be 'train', 'val', or 'test', got %r" % (split_type,))
    tw = windows[train_idx]
    if (len(train_idx) == 0):
        tw = windows
    mean = tw.mean(axis=(0, 1))
    std = tw.std(axis=(0, 1))
    std = np.where((std == 0.0), 1.0, std)
    wn = ((windows - mean) / std)
    if np.any(np.isnan(wn)):
        wn = np.nan_to_num(wn)
    return (wn[idx], wlabels[idx], mean, std)


def alfa_windows_from_points(sequence, label, w_size, stride):
    windows = []
    labels = []
    for i in range(0, ((len(sequence) - w_size) + 1), stride):
        window = sequence[i:(i + w_size)]
        windows.append(window)
        window_label = np.max(label[i:(i + w_size)])
        labels.append(window_label)
    if (len(windows) == 0):
        return (np.zeros((0, w_size, sequence.shape[1]), dtype=np.float64), np.array([], dtype=int))
    return (np.array(windows), np.array(labels))


def rfly_windows_from_points(sequence, label, w_size, stride):
    windows = []
    wlabels = []
    sz = int(((sequence.shape[0] - w_size) / stride))
    for i in range(0, sz):
        st = (i * stride)
        w = sequence[st:(st + w_size)]
        if (label[st:(st + w_size)].any() > 0):
            lbl = 1
        else:
            lbl = 0
        windows.append(w)
        wlabels.append(lbl)
    if (len(windows) == 0):
        return (np.zeros((0, w_size, sequence.shape[1]), dtype=np.float64), np.array([], dtype=int))
    return (np.stack(windows), np.stack(wlabels))


def get_injector(sample_batched, d_mean, d_std):
    sample_batched = ((sample_batched * d_std) + d_mean)
    injected_window = Injector(sample_batched)
    injected_window.injected_win = ((injected_window.injected_win - d_mean) / d_std)
    return injected_window.injected_win


def collate_test(batch):
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out


class ALFADataset(Dataset):

    def __init__(self, root_dir, subject_id, split_type='train', is_cuda=True, verbose=False, window_size=100, window_stride=1, seed=42, train_ratio=0.8, domain_source=True):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self.train_ratio = train_ratio
        self.domain_source = domain_source
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if ((random_choice == 0) and (len(self.negative) > 0)):
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)
        if (self.split_type == 'train'):
            sequence = self.apply_data_augmentation(sequence)
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {'sequence': sequence, 'sequence_mask': sequence_mask, 'positive': positive, 'negative': negative, 'label': label}
        return sample

    def apply_data_augmentation(self, sequence):
        augmented = sequence.copy()
        if (np.random.rand() < 0.5):
            noise = np.random.normal(0, 0.05, sequence.shape)
            augmented += noise
        if (np.random.rand() < 0.3):
            scale_factor = np.random.uniform(0.95, 1.05, sequence.shape[1])
            augmented = (augmented * scale_factor)
        if (np.random.rand() < 0.3):
            offset = int(np.random.uniform((- 5), 5))
            if (offset > 0):
                augmented[:(- offset)] = augmented[offset:]
                augmented[(- offset):] = augmented[((- offset) - 1):(- 1)]
            elif (offset < 0):
                offset = abs(offset)
                augmented[offset:] = augmented[:(- offset)]
                augmented[:offset] = augmented[1:(offset + 1)]
        if (np.random.rand() < 0.2):
            start = int(np.random.uniform(0, (sequence.shape[0] - 20)))
            end = (start + int(np.random.uniform(10, 20)))
            scale = np.random.uniform(0.8, 1.2)
            augmented[start:end] *= scale
        return augmented

    def load_sequence(self):
        data_path = os.path.join(self.root_dir, self.subject_id, 'data.csv')
        data_df = pd.read_csv(data_path)
        label_path = os.path.join(self.root_dir, self.subject_id, 'label.csv')
        label_df = pd.read_csv(label_path)
        raw_seq = data_df.iloc[:, 1:31].values.astype(float)
        raw_lab = label_df['failure_label'].values.astype(int)
        if self.verbose:
            print(f'ALFA Dataset {self.subject_id}:')
            print(f'  Features shape: {raw_seq.shape}')
            print(f'  Labels shape: {raw_lab.shape}')
            print(f'  Normal samples: {np.sum((raw_lab == 0))}')
            print(f'  Abnormal samples: {np.sum((raw_lab == 1))}')
        if np.any(np.isnan(raw_seq)):
            print(f'ALFA Dataset {self.subject_id}: Data contains NaN, replacing with zero')
            raw_seq = np.nan_to_num(raw_seq)
        (windows, wlabels) = alfa_windows_from_points(raw_seq, raw_lab, self.window_size, self.window_stride)
        (self.sequence, self.label, self.mean, self.std) = finalize_split_windows(windows, wlabels, self.split_type, self.domain_source, self.seed, self.train_ratio, self.subject_id, verbose=self.verbose, log_prefix=('ALFA %s' % self.subject_id))
        self.positive = self.sequence[(self.label == 0)]
        self.negative = self.sequence[(self.label == 1)]
        if self.verbose:
            print(f'  After windowing+split ({self.split_type}): {len(self.sequence)} windows')
            print(f'  Positive samples: {len(self.positive)}')
            print(f'  Negative samples: {len(self.negative)}')

    def get_statistic(self):
        return (self.mean, self.std)


class ALFADataset_trg(Dataset):

    def __init__(self, root_dir, subject_id, split_type='train', is_cuda=True, verbose=False, window_size=100, window_stride=1, seed=42, train_ratio=0.8, domain_source=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self.train_ratio = train_ratio
        self.domain_source = domain_source
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs((id_ - np.random.randint(1, 11)))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        if (negative.shape != sequence.shape):
            negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {'sequence': sequence, 'sequence_mask': sequence_mask, 'positive': positive, 'negative': negative, 'label': label}
        return sample

    def load_sequence(self):
        data_path = os.path.join(self.root_dir, self.subject_id, 'data.csv')
        data_df = pd.read_csv(data_path)
        label_path = os.path.join(self.root_dir, self.subject_id, 'label.csv')
        label_df = pd.read_csv(label_path)
        raw_seq = data_df.iloc[:, 1:31].values.astype(float)
        raw_lab = label_df['failure_label'].values.astype(int)
        if self.verbose:
            print(f'ALFA Target Dataset {self.subject_id}:')
            print(f'  Features shape: {raw_seq.shape}')
            print(f'  Labels shape: {raw_lab.shape}')
            print(f'  Normal samples: {np.sum((raw_lab == 0))}')
            print(f'  Abnormal samples: {np.sum((raw_lab == 1))}')
        if np.any(np.isnan(raw_seq)):
            print(f'ALFA Target Dataset {self.subject_id}: Data contains NaN, replacing with zero')
            raw_seq = np.nan_to_num(raw_seq)
        (windows, wlabels) = alfa_windows_from_points(raw_seq, raw_lab, self.window_size, self.window_stride)
        (self.sequence, self.label, self.mean, self.std) = finalize_split_windows(windows, wlabels, self.split_type, self.domain_source, self.seed, self.train_ratio, self.subject_id, verbose=self.verbose, log_prefix=('ALFA trg %s' % self.subject_id))
        self.positive = self.sequence[(self.label == 0)]
        self.negative = self.sequence[(self.label == 1)]
        if self.verbose:
            print(f'  After windowing+split ({self.split_type}): {len(self.sequence)} windows')
            print(f'  Positive samples: {len(self.positive)}')
            print(f'  Negative samples: {len(self.negative)}')

    def get_statistic(self):
        return (self.mean, self.std)


class RflyMADDataset(Dataset):

    def __init__(self, root_dir, subject_id, split_type='train', is_cuda=True, verbose=False, window_size=100, window_stride=1, seed=42, train_ratio=0.8, domain_source=True):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self.train_ratio = train_ratio
        self.domain_source = domain_source
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = np.random.randint(0, len(self.positive))
        positive = self.positive[pid_]
        random_choice = np.random.randint(0, 10)
        if ((random_choice == 0) and (len(self.negative) > 0)):
            nid_ = np.random.randint(0, len(self.negative))
            negative = self.negative[nid_]
        else:
            negative = get_injector(sequence, self.mean, self.std)
        if (self.split_type == 'train'):
            sequence = self.apply_data_augmentation(sequence)
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {'sequence': sequence, 'sequence_mask': sequence_mask, 'positive': positive, 'negative': negative, 'label': label}
        return sample

    def apply_data_augmentation(self, sequence):
        augmented = sequence.copy()
        if (np.random.rand() < 0.5):
            noise = np.random.normal(0, 0.05, sequence.shape)
            augmented += noise
        if (np.random.rand() < 0.3):
            scale_factor = np.random.uniform(0.95, 1.05, sequence.shape[1])
            augmented = (augmented * scale_factor)
        if (np.random.rand() < 0.3):
            offset = int(np.random.uniform((- 5), 5))
            if (offset > 0):
                augmented[:(- offset)] = augmented[offset:]
                augmented[(- offset):] = augmented[((- offset) - 1):(- 1)]
            elif (offset < 0):
                offset = abs(offset)
                augmented[offset:] = augmented[:(- offset)]
                augmented[:offset] = augmented[1:(offset + 1)]
        if (np.random.rand() < 0.2):
            start = int(np.random.uniform(0, (sequence.shape[0] - 20)))
            end = (start + int(np.random.uniform(10, 20)))
            scale = np.random.uniform(0.8, 1.2)
            augmented[start:end] *= scale
        return augmented

    def load_sequence(self):
        data_path = os.path.join(self.root_dir, self.subject_id, 'data.csv')
        data_df = pd.read_csv(data_path)
        label_path = os.path.join(self.root_dir, self.subject_id, 'label.csv')
        label_df = pd.read_csv(label_path)
        raw_seq = data_df.values.astype(float)
        raw_lab = label_df['label'].values.astype(int)
        if self.verbose:
            print(f'RflyMAD Dataset {self.subject_id}:')
            print(f'  Features shape: {raw_seq.shape}')
            print(f'  Labels shape: {raw_lab.shape}')
            print(f'  Normal samples: {np.sum((raw_lab == 0))}')
            print(f'  Abnormal samples: {np.sum((raw_lab == 1))}')
        if np.any(np.isnan(raw_seq)):
            print(f'RflyMAD Dataset {self.subject_id}: Data contains NaN, replacing with zero')
            raw_seq = np.nan_to_num(raw_seq)
        (windows, wlabels) = rfly_windows_from_points(raw_seq, raw_lab, self.window_size, self.window_stride)
        (self.sequence, self.label, self.mean, self.std) = finalize_split_windows(windows, wlabels, self.split_type, self.domain_source, self.seed, self.train_ratio, self.subject_id, verbose=self.verbose, log_prefix=('RflyMAD %s' % self.subject_id))
        self.positive = self.sequence[(self.label == 0)]
        self.negative = self.sequence[(self.label == 1)]
        if self.verbose:
            print(f'  After windowing+split ({self.split_type}): {len(self.sequence)} windows')
            print(f'  Positive samples: {len(self.positive)}')
            print(f'  Negative samples: {len(self.negative)}')

    def get_statistic(self):
        return (self.mean, self.std)


class RflyMADDataset_trg(Dataset):

    def __init__(self, root_dir, subject_id, split_type='train', is_cuda=True, verbose=False, window_size=100, window_stride=1, seed=42, train_ratio=0.8, domain_source=False):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.window_size = window_size
        self.window_stride = window_stride
        self.seed = seed
        self.train_ratio = train_ratio
        self.domain_source = domain_source
        self.load_sequence()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, id_):
        sequence = self.sequence[id_]
        pid_ = abs((id_ - np.random.randint(1, 11)))
        positive = self.sequence[pid_]
        self.positive = positive
        negative = get_injector(sequence, self.mean, self.std)
        if (negative.shape != sequence.shape):
            negative = get_injector(sequence, self.mean, self.std)
        self.negative = negative
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            positive = torch.Tensor(positive).float().cuda()
            negative = torch.Tensor(negative).float().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            positive = torch.Tensor(positive).float()
            negative = torch.Tensor(negative).float()
            label = torch.Tensor([label]).long()
        sample = {'sequence': sequence, 'sequence_mask': sequence_mask, 'positive': positive, 'negative': negative, 'label': label}
        return sample

    def load_sequence(self):
        data_path = os.path.join(self.root_dir, self.subject_id, 'data.csv')
        data_df = pd.read_csv(data_path)
        label_path = os.path.join(self.root_dir, self.subject_id, 'label.csv')
        label_df = pd.read_csv(label_path)
        raw_seq = data_df.values.astype(float)
        raw_lab = label_df['label'].values.astype(int)
        if self.verbose:
            print(f'RflyMAD Target Dataset {self.subject_id}:')
            print(f'  Features shape: {raw_seq.shape}')
            print(f'  Labels shape: {raw_lab.shape}')
            print(f'  Normal samples: {np.sum((raw_lab == 0))}')
            print(f'  Abnormal samples: {np.sum((raw_lab == 1))}')
        if np.any(np.isnan(raw_seq)):
            print(f'RflyMAD Target Dataset {self.subject_id}: Data contains NaN, replacing with zero')
            raw_seq = np.nan_to_num(raw_seq)
        (windows, wlabels) = rfly_windows_from_points(raw_seq, raw_lab, self.window_size, self.window_stride)
        (self.sequence, self.label, self.mean, self.std) = finalize_split_windows(windows, wlabels, self.split_type, self.domain_source, self.seed, self.train_ratio, self.subject_id, verbose=self.verbose, log_prefix=('RflyMAD trg %s' % self.subject_id))
        self.positive = self.sequence[(self.label == 0)]
        self.negative = self.sequence[(self.label == 1)]
        if self.verbose:
            print(f'  After windowing+split ({self.split_type}): {len(self.sequence)} windows')

    def get_statistic(self):
        return (self.mean, self.std)
