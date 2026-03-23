import logging
import math
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, precision_recall_curve

def get_logger(log_file):
    """Dedicated logger per log file (no root pollution) — avoids duplicate console lines when train + eval both run."""
    log_abs = os.path.abspath(log_file)
    log_dir = os.path.dirname(log_abs)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    name = 'protoda.' + log_abs
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_abs, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    def log(s):
        logger.info(s)
    return log


def format_scores_line(metrics_pred, *, full=False):
    prc = metrics_pred.get('avg_prc', float('nan'))
    f1v = metrics_pred.get('best_f1', float('nan'))
    if math.isnan(prc) or math.isnan(f1v):
        return 'N/A (no positive class or empty eval)'
    if full:
        roc = metrics_pred.get('roc_auc', float('nan'))
        prec = metrics_pred.get('best_prec', float('nan'))
        rec = metrics_pred.get('best_rec', float('nan'))
        return ('AUPRC=%.4f  AUROC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f' % (prc, roc, f1v, prec, rec))
    return ('AUPRC=%.4f  F1=%.4f' % (prc, f1v))
dict_metrics = {'alfa': {'acc': accuracy_score, 'mac_f1': f1_score, 'w_f1': f1_score}, 'rflymad': {'acc': accuracy_score, 'mac_f1': f1_score, 'w_f1': f1_score}}

def get_dataset_type(args):
    d = getattr(args, 'dataset', None)
    if (d in ('alfa', 'rflymad')):
        return d
    raise ValueError('dataset must be alfa or rflymad, got %r' % (d,))

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = (self.sum / self.count)

    def __str__(self):
        fmtstr = (((('{name} {val' + self.fmt) + '} ({avg') + self.fmt) + '})')
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, is_logged=False):
        entries = [(self.prefix + self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        line = '  |  '.join(entries)
        if (not is_logged):
            print(line)
        else:
            return line

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str((num_batches // 1)))
        fmt = (('{:' + str(num_digits)) + 'd}')
        return (((('[' + fmt) + '/') + fmt.format(num_batches)) + ']')

class PredictionMeter(object):

    def __init__(self, args):
        self.args = args
        self.dataset_type = get_dataset_type(args)
        self.target_list = []
        self.output_list = []
        self.id_patient_list = []
        self.stay_hours_list = []
        self.dict_metrics = {}
        self.dict_metrics = dict_metrics.get(self.dataset_type, {})

    def update(self, target, output, id_patient=None, stay_hour=None):
        output_np = output.detach().cpu().numpy().flatten()
        target_np = target.detach().cpu().numpy().flatten()
        self.output_list = (self.output_list + list(output_np))
        self.target_list = (self.target_list + list(target_np))
        if (id_patient is not None):
            id_patient_np = id_patient.numpy().flatten()
            self.id_patient_list = (self.id_patient_list + list(id_patient_np))
        if (stay_hour is not None):
            stay_hour_np = stay_hour.numpy().flatten()
            self.stay_hours_list = (self.stay_hours_list + list(stay_hour_np))

    def get_metrics(self):
        return_dict = {}
        output = np.array(self.output_list)
        target = np.array(self.target_list)
        if ((len(output) == 0) or (len(target) == 0)):
            return_dict['best_f1'] = float('nan')
            return_dict['best_prec'] = float('nan')
            return_dict['best_rec'] = float('nan')
            return_dict['best_thr'] = float('nan')
            return_dict['avg_prc'] = float('nan')
            return_dict['roc_auc'] = float('nan')
            return_dict['mac_f1'] = float('nan')
            return_dict['w_f1'] = float('nan')
            return_dict['acc'] = float('nan')
            return return_dict
        unique_labels = np.unique(target)
        if (len(unique_labels) < 2):
            return_dict['best_f1'] = float('nan')
            return_dict['best_prec'] = float('nan')
            return_dict['best_rec'] = float('nan')
            return_dict['best_thr'] = float('nan')
            return_dict['avg_prc'] = float('nan')
            return_dict['roc_auc'] = float('nan')
            return_dict['mac_f1'] = float('nan')
            return_dict['w_f1'] = float('nan')
            return_dict['acc'] = float('nan')
            return return_dict
        avg_prc = average_precision_score(target, output, pos_label=1)
        roc_auc = 0
        try:
            roc_auc = roc_auc_score(target, output)
        except ValueError:
            roc_auc = float('nan')
        try:
            (prec, rec, thr) = precision_recall_curve(target, output, pos_label=1)
            prec = np.where(np.isnan(prec), 0.0, prec)
            rec = np.where(np.isnan(rec), 0.0, rec)
            with np.errstate(invalid='ignore'):
                f1score = np.where(((rec + prec) == 0.0), 0.0, (((2 * prec) * rec) / (rec + prec)))
            best_f1_index = np.argmax(f1score)
            return_dict['best_f1'] = f1score[best_f1_index]
            return_dict['best_prec'] = prec[best_f1_index]
            return_dict['best_rec'] = rec[best_f1_index]
            return_dict['best_thr'] = thr[best_f1_index]
            return_dict['avg_prc'] = avg_prc
            return_dict['roc_auc'] = roc_auc
            output_binary = np.where((output[:] > thr[best_f1_index]), 1, 0)
            return_dict['mac_f1'] = f1_score(target, output_binary, average='macro')
            return_dict['w_f1'] = f1_score(target, output_binary, average='weighted')
            return_dict['acc'] = accuracy_score(target, output_binary)
        except Exception as e:
            return_dict['best_f1'] = float('nan')
            return_dict['best_prec'] = float('nan')
            return_dict['best_rec'] = float('nan')
            return_dict['best_thr'] = float('nan')
            return_dict['avg_prc'] = float('nan')
            return_dict['roc_auc'] = roc_auc
            return_dict['mac_f1'] = float('nan')
            return_dict['w_f1'] = float('nan')
            return_dict['acc'] = float('nan')
        return return_dict

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        (_, pred) = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, (- 1)).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape((- 1)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_((100.0 / batch_size)))
        return res
