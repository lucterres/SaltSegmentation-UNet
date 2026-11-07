import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from utils import config
from utils.dataset import get_transforms
from utils.model import UNet

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
BASE_CHECKPOINT = Path(
    'resultsHPC/scenario_A_seed42_nreal1200_train/best_model.pth'
)
AUGMENTED_CHECKPOINT = Path(
    'resultsHPC/scenario_B_seed42_nreal1200_train_pairs1600_seismic_ns1200/best_model.pth'
)


@dataclass(frozen=True)
class TestPair:
    image_id: str
    image_path: Path
    mask_path: Path


def default_test_dir(platform_name=None):
    platform_name = platform_name or os.name
    if platform_name == 'nt':
        return Path(r'D:\dataset\tgs-salt\subset_split\test')
    return Path('/var/tmp/cym7/datasets/subset_split/test')


def resolve_test_dirs(test_dir):
    test_dir = Path(test_dir)
    if test_dir.name.lower() == 'images':
        return test_dir, test_dir.parent / 'masks'
    if test_dir.name.lower() == 'masks':
        return test_dir.parent / 'images', test_dir
    return test_dir / 'images', test_dir / 'masks'


def _paths_by_id(directory, label):
    if not directory.is_dir():
        raise FileNotFoundError(f'{label} directory not found: {directory}')

    paths = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in paths:
            raise ValueError(f'duplicate {label} ID: {path.stem}')
        paths[path.stem] = path
    return paths


def build_pairs(test_dir):
    image_dir, mask_dir = resolve_test_dirs(test_dir)
    images = _paths_by_id(image_dir, 'image')
    masks = _paths_by_id(mask_dir, 'mask')
    image_ids = set(images)
    mask_ids = set(masks)
    if image_ids != mask_ids:
        missing_masks = sorted(image_ids - mask_ids)
        missing_images = sorted(mask_ids - image_ids)
        raise ValueError(
            'image/mask IDs do not match: '
            f'missing masks={missing_masks[:5]}, missing images={missing_images[:5]}'
        )
    if not images:
        raise ValueError(f'no image/mask IDs found in {test_dir}')

    return [
        TestPair(image_id, images[image_id], masks[image_id])
        for image_id in sorted(image_ids)
    ]


def load_pair(image_path, mask_path, image_transform=None, mask_transform=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'could not read image: {image_path}')
    if mask is None:
        raise ValueError(f'could not read mask: {mask_path}')

    if image_transform is None or mask_transform is None:
        image_transform, mask_transform = get_transforms(
            config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH
        )
    image = image_transform(image)
    mask = (mask_transform(mask) > 0.5).float()
    return image, mask


class PairedTestDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.image_transform, self.mask_transform = get_transforms(
            config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        image, mask = load_pair(
            pair.image_path,
            pair.mask_path,
            self.image_transform,
            self.mask_transform,
        )
        return image, mask


def compute_per_image_metrics(logits, targets, threshold=config.THRESHOLD):
    predictions = torch.sigmoid(logits) > threshold
    targets = targets > 0.5
    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    predicted_count = predictions.sum(dim=(1, 2, 3)).float()
    target_count = targets.sum(dim=(1, 2, 3)).float()
    union = predicted_count + target_count - intersection
    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (predicted_count + target_count + 1e-8)
    return [
        {'iou': float(iou_value), 'dice': float(dice_value)}
        for iou_value, dice_value in zip(iou.cpu(), dice.cpu())
    ]


def load_model(checkpoint_path, device='cpu'):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    device = torch.device(device)
    model = UNet()
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, TypeError, AttributeError) as error:
        raise ValueError(f'incompatible checkpoint: {checkpoint_path}') from error
    return model.to(device).eval()


def evaluate_models(base_model, augmented_model, pairs, batch_size=16,
                    device='cpu', threshold=config.THRESHOLD):
    device = torch.device(device)
    pairs = sorted(pairs, key=lambda pair: pair.image_id)
    loader = DataLoader(
        PairedTestDataset(pairs), batch_size=batch_size, shuffle=False
    )
    base_model = base_model.to(device).eval()
    augmented_model = augmented_model.to(device).eval()
    rows = []

    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            base_metrics = compute_per_image_metrics(
                base_model(images), masks, threshold
            )
            augmented_metrics = compute_per_image_metrics(
                augmented_model(images), masks, threshold
            )
            first_index = batch_index * batch_size
            batch_pairs = pairs[first_index:first_index + len(base_metrics)]
            for pair, base, augmented, mask in zip(
                batch_pairs, base_metrics, augmented_metrics, masks
            ):
                rows.append({
                    'image_id': pair.image_id,
                    'iou_base': base['iou'],
                    'iou_augmented': augmented['iou'],
                    'iou_difference': augmented['iou'] - base['iou'],
                    'dice_base': base['dice'],
                    'dice_augmented': augmented['dice'],
                    'dice_difference': augmented['dice'] - base['dice'],
                    'ground_truth_empty': not bool((mask > 0.5).any().item()),
                })
    return rows


def write_per_image_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'image_id', 'iou_base', 'iou_augmented', 'iou_difference',
        'dice_base', 'dice_augmented', 'dice_difference', 'ground_truth_empty',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['image_id']))
    return output_path


def compute_paired_statistics(iou_base, iou_augmented, alternative='greater',
                              alpha=0.05):
    base = np.asarray(iou_base, dtype=float)
    augmented = np.asarray(iou_augmented, dtype=float)
    if base.ndim != 1 or augmented.ndim != 1 or base.size != augmented.size:
        raise ValueError('base and augmented inputs must have the same number of IoU values')
    if base.size == 0:
        raise ValueError('at least one paired IoU value is required')
    if not np.isfinite(base).all() or not np.isfinite(augmented).all():
        raise ValueError('IoU values must be finite')
    if ((base < 0) | (base > 1)).any() or ((augmented < 0) | (augmented > 1)).any():
        raise ValueError('IoU values must be within [0, 1]')
    if alternative not in {'greater', 'less', 'two-sided'}:
        raise ValueError(f'unsupported Wilcoxon alternative: {alternative}')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')

    differences = augmented - base
    nonzero = differences[differences != 0]
    zero_count = int(differences.size - nonzero.size)
    if nonzero.size == 0:
        statistic = 0.0
        p_value = 1.0
        positive_rank_sum = 0.0
        negative_rank_sum = 0.0
        rank_biserial = None
    else:
        statistic_result = stats.wilcoxon(
            augmented,
            base,
            alternative=alternative,
            zero_method='wilcox',
            method='approx',
        )
        statistic = float(statistic_result.statistic)
        p_value = float(statistic_result.pvalue)
        ranks = stats.rankdata(np.abs(nonzero), method='average')
        positive_rank_sum = float(ranks[nonzero > 0].sum())
        negative_rank_sum = float(ranks[nonzero < 0].sum())
        rank_total = positive_rank_sum + negative_rank_sum
        rank_biserial = (
            float((positive_rank_sum - negative_rank_sum) / rank_total)
            if rank_total else None
        )

    diagnostic_reason = None
    shapiro_w = shapiro_p = skewness = None
    if differences.size < 3:
        diagnostic_reason = 'fewer than three paired differences'
    elif np.all(differences == differences[0]):
        diagnostic_reason = 'constant paired differences'
    else:
        shapiro_result = stats.shapiro(differences)
        shapiro_w = float(shapiro_result.statistic)
        shapiro_p = float(shapiro_result.pvalue)
        skewness = float(stats.skew(differences, bias=True))

    return {
        'descriptive': {
            'base_iou_median': float(np.median(base)),
            'base_iou_iqr': float(stats.iqr(base)),
            'augmented_iou_median': float(np.median(augmented)),
            'augmented_iou_iqr': float(stats.iqr(augmented)),
            'difference_median': float(np.median(differences)),
            'favorable_pair_percentage': float(
                100.0 * np.count_nonzero(differences > 0) / differences.size
            ),
        },
        'wilcoxon': {
            'alternative': alternative,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': float(alpha),
            'significant': bool(p_value < alpha),
            'positive_rank_sum': positive_rank_sum,
            'negative_rank_sum': negative_rank_sum,
            'zero_difference_count': zero_count,
            'nonzero_pair_count': int(nonzero.size),
            'zero_method': 'wilcox',
            'rank_tie_method': 'average',
            'p_value_method': 'approx',
        },
        'effect_size': {'rank_biserial': rank_biserial},
        'diagnostics': {
            'shapiro_w': shapiro_w,
            'shapiro_p_value': shapiro_p,
            'skewness': skewness,
            'unavailable_reason': diagnostic_reason,
        },
    }


def write_summary_json(statistics, settings, rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(row['image_id'] for row in rows)
    id_digest = hashlib.sha256('\n'.join(image_ids).encode('utf-8')).hexdigest()
    base_dice = np.asarray([row['dice_base'] for row in rows], dtype=float)
    augmented_dice = np.asarray([row['dice_augmented'] for row in rows], dtype=float)
    summary = {
        'statistics': statistics,
        'effective_settings': settings,
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'opencv': cv2.__version__,
        },
        'data_integrity': {
            'validated_pair_count': len(rows),
            'empty_ground_truth_count': sum(
                bool(row['ground_truth_empty']) for row in rows
            ),
        },
        'tested_ids_sha256': id_digest,
        'dice_descriptive': {
            'base_median': float(np.median(base_dice)),
            'base_iqr': float(stats.iqr(base_dice)),
            'augmented_median': float(np.median(augmented_dice)),
            'augmented_iqr': float(stats.iqr(augmented_dice)),
        },
        'metric_convention': 'intersection / (union + 1e-8); empty union => IoU 0',
    }
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)
        output_file.write('\n')
    return output_path


def write_diagnostic_plots(rows, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=lambda row: row['image_id'])
    base = [row['iou_base'] for row in ordered_rows]
    augmented = [row['iou_augmented'] for row in ordered_rows]
    differences = [row['iou_difference'] for row in ordered_rows]

    paired_path = output_dir / 'paired_iou.png'
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(base, augmented, alpha=0.55, s=18)
    axis.plot([0, 1], [0, 1], color='black', linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Base IoU', ylabel='Augmented IoU')
    axis.set_title('Per-image paired IoU')
    figure.tight_layout()
    figure.savefig(str(paired_path), dpi=140)
    plt.close(figure)

    difference_path = output_dir / 'iou_difference.png'
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(differences, bins=30)
    axis.axvline(0, color='black', linewidth=1)
    axis.set(xlabel='Augmented IoU - Base IoU', ylabel='Image count')
    axis.set_title('Per-image IoU differences')
    figure.tight_layout()
    figure.savefig(str(difference_path), dpi=140)
    plt.close(figure)
    return paired_path, difference_path


def _threshold_value(value):
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('threshold must be a number') from error
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError('threshold must be within [0, 1]')
    return threshold


def _alpha_value(value):
    try:
        alpha = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('alpha must be a number') from error
    if not 0 < alpha < 1:
        raise argparse.ArgumentTypeError('alpha must be between 0 and 1')
    return alpha


def _positive_integer(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('batch size must be a positive integer') from error
    if number <= 0:
        raise argparse.ArgumentTypeError('batch size must be a positive integer')
    return number


def _device_value(value):
    if value in {'auto', 'cpu', 'cuda'}:
        return value
    if value.startswith('cuda:') and value[5:].isdigit():
        return value
    raise argparse.ArgumentTypeError('device must be auto, cpu, cuda, or cuda:N')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Paired per-image IoU test')
    parser.add_argument('--base-checkpoint', default=str(BASE_CHECKPOINT))
    parser.add_argument('--aug-checkpoint', default=str(AUGMENTED_CHECKPOINT))
    parser.add_argument('--test-dir', default=str(default_test_dir()))
    parser.add_argument('--output-dir', default='resultsHPC/paired_iou_test_a')
    parser.add_argument('--threshold', type=_threshold_value, default=config.THRESHOLD)
    parser.add_argument('--alpha', type=_alpha_value, default=0.05)
    parser.add_argument(
        '--alternative', choices=('greater', 'less', 'two-sided'), default='greater'
    )
    parser.add_argument('--batch-size', type=_positive_integer, default=config.BATCH_SIZE)
    parser.add_argument('--device', type=_device_value, default='auto')
    return parser.parse_args(argv)


def validate_inputs(args):
    base_checkpoint = Path(args.base_checkpoint)
    augmented_checkpoint = Path(args.aug_checkpoint)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f'base checkpoint not found: {base_checkpoint}')
    if not augmented_checkpoint.is_file():
        raise FileNotFoundError(f'augmented checkpoint not found: {augmented_checkpoint}')

    pairs = build_pairs(args.test_dir)
    canonical_dir = default_test_dir().resolve()
    if Path(args.test_dir).resolve() == canonical_dir and len(pairs) != 800:
        raise ValueError(
            f'canonical test directory must contain exactly 800 pairs, found {len(pairs)}'
        )
    return pairs


def resolve_device(device):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return resolved


def main(argv=None):
    args = parse_args(argv)
    try:
        device = resolve_device(args.device)
        pairs = validate_inputs(args)
        base_model = load_model(args.base_checkpoint, device)
        augmented_model = load_model(args.aug_checkpoint, device)
        rows = evaluate_models(
            base_model,
            augmented_model,
            pairs,
            batch_size=args.batch_size,
            device=device,
            threshold=args.threshold,
        )
        statistics = compute_paired_statistics(
            [row['iou_base'] for row in rows],
            [row['iou_augmented'] for row in rows],
            alternative=args.alternative,
            alpha=args.alpha,
        )

        output_dir = Path(args.output_dir)
        settings = {
            'base_checkpoint': str(Path(args.base_checkpoint)),
            'augmented_checkpoint': str(Path(args.aug_checkpoint)),
            'test_dir': str(Path(args.test_dir)),
            'output_dir': str(output_dir),
            'threshold': args.threshold,
            'alpha': args.alpha,
            'alternative': args.alternative,
            'batch_size': args.batch_size,
            'device': str(device),
        }
        write_per_image_csv(rows, output_dir / 'per_image_metrics.csv')
        write_summary_json(statistics, settings, rows, output_dir / 'summary.json')
        write_diagnostic_plots(rows, output_dir)

        wilcoxon = statistics['wilcoxon']
        print(f"[RESULT] paired images: {len(rows)}")
        print(f"[RESULT] Wilcoxon W={wilcoxon['statistic']:.4f} p={wilcoxon['p_value']:.6g}")
        print(f"[RESULT] rank-biserial r={statistics['effect_size']['rank_biserial']}")
        print(f'[INFO] Outputs: {output_dir}')
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f'[ERROR] {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    main()
