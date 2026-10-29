import csv
import os
import string
import sys

from rapidfuzz.distance import Levenshtein

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


class LengthErrorMetric(object):
    """Rec metric with short / medium / long buckets and edit analysis."""

    def __init__(
        self,
        main_indicator='acc',
        is_filter=False,
        ignore_space=True,
        short_max=5,
        medium_max=10,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.short_max = int(short_max)
        self.medium_max = int(medium_max)
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def _prepare_pair(self, pred, target):
        if self.ignore_space:
            pred = pred.replace(' ', '')
            target = target.replace(' ', '')
        if self.is_filter:
            pred = self._normalize_text(pred)
            target = self._normalize_text(target)
        return pred, target

    def _bucket_name(self, target):
        target_len = len(target)
        if target_len <= self.short_max:
            return 'short'
        if target_len <= self.medium_max:
            return 'medium'
        return 'long'

    def _empty_stats(self):
        return {
            'samples': 0,
            'correct': 0,
            'edit_distance': 0,
            'ref_chars': 0,
            'pred_chars': 0,
            'len_abs_diff': 0,
            'norm_edit_dis': 0.0,
            'substitutions': 0,
            'insertions': 0,
            'deletions': 0,
        }

    def _add_pair(self, stats, pred, target):
        editops = Levenshtein.editops(pred, target)
        substitutions = sum(1 for op in editops if op.tag == 'replace')
        deletions = sum(1 for op in editops if op.tag == 'delete')
        insertions = sum(1 for op in editops if op.tag == 'insert')
        dist = substitutions + deletions + insertions

        stats['samples'] += 1
        stats['correct'] += int(pred == target)
        stats['edit_distance'] += dist
        stats['ref_chars'] += len(target)
        stats['pred_chars'] += len(pred)
        stats['len_abs_diff'] += abs(len(pred) - len(target))
        stats['norm_edit_dis'] += Levenshtein.normalized_distance(pred, target)
        stats['substitutions'] += substitutions
        stats['insertions'] += insertions
        stats['deletions'] += deletions

    def _select_pred_label(self, pred_label):
        # GTC-style postprocess returns [gtc_result, ctc_result], where each
        # branch is the usual (preds, labels) pair. Pick the requested branch.
        if isinstance(pred_label, list) and len(pred_label) == 2:
            if all(
                    isinstance(branch, (list, tuple)) and len(branch) == 2
                    for branch in pred_label):
                if self.main_indicator.startswith('gtc_'):
                    return pred_label[0]
                return pred_label[1]
        return pred_label

    def __call__(self, pred_label, *args, **kwargs):
        pred_label = self._select_pred_label(pred_label)
        preds, labels = pred_label
        for (pred, _), (target, _) in zip(preds, labels):
            pred, target = self._prepare_pair(pred, target)
            self._add_pair(self.total, pred, target)
            self._add_pair(self.buckets[self._bucket_name(target)], pred, target)

        return self._format_metric(self.total, prefix='')

    def _format_metric(self, stats, prefix):
        samples = stats['samples']
        ref_chars = stats['ref_chars']
        cer = stats['edit_distance'] / (ref_chars + self.eps)
        return {
            f'{prefix}n': samples,
            f'{prefix}acc': stats['correct'] / (samples + self.eps),
            f'{prefix}cer': cer,
            f'{prefix}char_acc': max(0.0, 1.0 - cer),
            f'{prefix}norm_edit_dis': 1 - stats['norm_edit_dis'] / (samples + self.eps),
            f'{prefix}avg_len_abs_diff': stats['len_abs_diff'] / (samples + self.eps),
            f'{prefix}substitutions': stats['substitutions'],
            f'{prefix}insertions': stats['insertions'],
            f'{prefix}deletions': stats['deletions'],
        }

    def get_metric(self):
        metric = self._format_metric(self.total, prefix='')
        for name in ['short', 'medium', 'long']:
            metric.update(self._format_metric(self.buckets[name], prefix=f'{name}_'))
        self.reset()
        return metric

    def reset(self):
        self.total = self._empty_stats()
        self.buckets = {
            'short': self._empty_stats(),
            'medium': self._empty_stats(),
            'long': self._empty_stats(),
        }


def iter_paper_metrics(metric):
    # Keep only the metrics that are easy to compare in a paper table.
    keep_keys = [
        'n',
        'acc',
        'cer',
        'norm_edit_dis',
        'short_acc',
        'medium_acc',
        'long_acc',
        'fps',
    ]
    for key in keep_keys:
        if key in metric:
            yield key, metric[key]

def get_method_from_config(config_path):
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    return os.path.basename(config_dir)


def append_result_csv(metric, method, csv_path='result.csv'):
    fieldnames = [
        'method',
        'n',
        'acc',
        'cer',
        'norm_edit_dis',
        'short_acc',
        'medium_acc',
        'long_acc',
        'fps',
    ]

    row = {
        'method': method,
        'n': metric.get('n', ''),
        'acc': metric.get('acc', ''),
        'cer': metric.get('cer', ''),
        'norm_edit_dis': metric.get('norm_edit_dis', ''),
        'short_acc': metric.get('short_acc', ''),
        'medium_acc': metric.get('medium_acc', ''),
        'long_acc': metric.get('long_acc', ''),
        'fps': metric.get('fps', ''),
    }

    # 浮点数统一保留 5 位
    for key in [
        'acc',
        'cer',
        'norm_edit_dis',
        'short_acc',
        'medium_acc',
        'long_acc',
        'fps',
    ]:
        value = row[key]
        if isinstance(value, (float, int)) and value != '':
            row[key] = f'{value:.5f}'

    file_exists = os.path.exists(csv_path)
    file_empty = (not file_exists) or os.path.getsize(csv_path) == 0

    # --------------------------------------------------
    # 关键：如果旧 CSV 最后一行没有换行，先补换行
    # --------------------------------------------------
    if file_exists and not file_empty:
        with open(csv_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            last_char = f.read(1)

            if last_char not in (b'\n', b'\r'):
                f.seek(0, os.SEEK_END)
                f.write(b'\n')

    # --------------------------------------------------
    # 追加新结果
    # --------------------------------------------------
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if file_empty:
            writer.writeheader()

        writer.writerow(row)

    print(f'method: {method}')
    print(f'Result appended to: {csv_path}')

def parse_args():
    parser = ArgsParser()
    return parser.parse_args()


def main():
    flags = parse_args()

    config_path = flags.config
    cfg = Config(config_path)

    flag_dict = vars(flags)
    opt = flag_dict.pop('opt')
    cfg.merge_dict(flag_dict)
    cfg.merge_dict(opt)

    trainer = Trainer(cfg, mode='eval')
    trainer.eval_class = LengthErrorMetric(**cfg.cfg.get('Metric', {}))

    metric = trainer.eval()

    print('metric eval with length and error analysis ***************')

    for k, v in iter_paper_metrics(metric):
        if k == 'n':
            print(f'{k}:{v}')
        elif isinstance(v, float):
            print(f'{k}:{v:.5f}')
        else:
            print(f'{k}:{v}')

    # config.yml 的父目录名作为方法名
    method = get_method_from_config(config_path)

    append_result_csv(
        metric=metric,
        method=method,
        csv_path='result.csv',
    )

if __name__ == '__main__':
    main()
