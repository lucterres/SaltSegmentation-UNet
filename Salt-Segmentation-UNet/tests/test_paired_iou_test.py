import csv
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import paired_iou_test as cli
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
CLI_MODULE = PROJECT_DIR / 'paired_iou_test.py'


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, images):
        return torch.full_like(images, self.logit)


class TestPairedIoUTestModule(unittest.TestCase):
    def test_cli_module_exists(self):
        self.assertTrue(CLI_MODULE.is_file(), f'CLI module not found: {CLI_MODULE}')

    def require_helper(self, name):
        self.assertTrue(hasattr(cli, name), f'CLI helper not implemented: {name}')
        return getattr(cli, name)

    def test_default_test_directory_is_canonical_for_each_platform(self):
        default_test_dir = self.require_helper('default_test_dir')

        self.assertEqual(
            Path(default_test_dir('nt')),
            Path(r'D:\dataset\tgs-salt\subset_split\test'),
        )
        self.assertEqual(
            Path(default_test_dir('posix')),
            Path('/var/tmp/cym7/datasets/subset_split/test'),
        )

    def test_build_pairs_sorts_ids_and_rejects_incomplete_sets(self):
        build_pairs = self.require_helper('build_pairs')

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / 'images'
            masks = root / 'masks'
            images.mkdir()
            masks.mkdir()
            for image_id in ('zeta', 'alpha'):
                (images / f'{image_id}.png').touch()
                (masks / f'{image_id}.png').touch()

            pairs = build_pairs(root)
            self.assertEqual([pair.image_id for pair in pairs], ['alpha', 'zeta'])

            (masks / 'orphan.png').touch()
            with self.assertRaisesRegex(ValueError, 'image/mask IDs'):
                build_pairs(root)

    def test_build_pairs_rejects_duplicate_image_ids(self):
        build_pairs = self.require_helper('build_pairs')

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / 'images').mkdir()
            (root / 'masks').mkdir()
            (root / 'images' / 'duplicate.png').touch()
            (root / 'images' / 'duplicate.jpg').touch()
            (root / 'masks' / 'duplicate.png').touch()

            with self.assertRaisesRegex(ValueError, 'duplicate image ID'):
                build_pairs(root)

    def test_load_pair_resizes_grayscale_and_keeps_masks_binary(self):
        load_pair = self.require_helper('load_pair')

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / 'image.png'
            mask_path = root / 'mask.png'
            image = torch.arange(16, dtype=torch.uint8).reshape(4, 4).numpy()
            mask = torch.zeros((4, 4), dtype=torch.uint8).numpy()
            mask[1:3, 1:3] = 255
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            image_tensor, mask_tensor = load_pair(image_path, mask_path)

            self.assertEqual(tuple(image_tensor.shape), (1, 128, 128))
            self.assertEqual(tuple(mask_tensor.shape), (1, 128, 128))
            self.assertGreaterEqual(float(image_tensor.min()), 0.0)
            self.assertLessEqual(float(image_tensor.max()), 1.0)
            self.assertEqual(set(mask_tensor.unique().tolist()), {0.0, 1.0})

    def test_compute_per_image_metrics_preserves_rows_and_empty_union_policy(self):
        compute_metrics = self.require_helper('compute_per_image_metrics')
        logits = torch.tensor([
            [[[5.0, -5.0], [-5.0, 5.0]]],
            [[[-5.0, -5.0], [-5.0, -5.0]]],
        ])
        targets = torch.tensor([
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ])

        metrics = compute_metrics(logits, targets, threshold=0.5)

        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0], {'iou': 1.0, 'dice': 1.0})
        self.assertEqual(metrics[1], {'iou': 0.0, 'dice': 0.0})

    def test_load_model_restores_a_state_dict_on_the_requested_device(self):
        load_model = self.require_helper('load_model')

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / 'weights.pth'
            original = cli.UNet()
            torch.save(original.state_dict(), checkpoint_path)

            restored = load_model(checkpoint_path, device='cpu')

            self.assertIsInstance(restored, cli.UNet)
            for original_value, restored_value in zip(
                original.state_dict().values(), restored.state_dict().values()
            ):
                self.assertTrue(torch.equal(original_value, restored_value))

    def test_load_model_rejects_incompatible_state_dict(self):
        load_model = self.require_helper('load_model')

        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / 'invalid.pth'
            torch.save({'unexpected.weight': torch.ones(1)}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, 'incompatible checkpoint'):
                load_model(checkpoint_path, device='cpu')

    def test_evaluate_models_returns_per_image_paired_metrics(self):
        evaluate_models = self.require_helper('evaluate_models')

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / 'sample.png'
            mask_path = root / 'sample-mask.png'
            image = torch.zeros((4, 4), dtype=torch.uint8).numpy()
            mask = torch.full((4, 4), 255, dtype=torch.uint8).numpy()
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            pair = cli.TestPair('sample', image_path, mask_path)

            rows = evaluate_models(
                FixedLogitModel(10.0),
                FixedLogitModel(-10.0),
                [pair],
                batch_size=1,
                device='cpu',
                threshold=0.5,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['image_id'], 'sample')
        self.assertEqual(rows[0]['iou_base'], 1.0)
        self.assertEqual(rows[0]['iou_augmented'], 0.0)
        self.assertEqual(rows[0]['iou_difference'], -1.0)
        self.assertFalse(rows[0]['ground_truth_empty'])

    def test_per_image_csv_is_sorted_by_image_id(self):
        write_per_image_csv = self.require_helper('write_per_image_csv')
        rows = [
            {'image_id': 'zeta', 'iou_base': 0.2},
            {'image_id': 'alpha', 'iou_base': 0.8},
        ]

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / 'per_image_metrics.csv'
            write_per_image_csv(rows, output_path)
            with output_path.open(newline='') as output_file:
                saved_rows = list(csv.DictReader(output_file))

        self.assertEqual([row['image_id'] for row in saved_rows], ['alpha', 'zeta'])

    def test_paired_statistics_detect_one_sided_positive_change(self):
        analyze = self.require_helper('compute_paired_statistics')
        base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        augmented = [0.2, 0.4, 0.35, 0.55, 0.7, 0.9]

        summary = analyze(base, augmented, alternative='greater', alpha=0.05)

        self.assertEqual(summary['wilcoxon']['alternative'], 'greater')
        self.assertLess(summary['wilcoxon']['p_value'], 0.05)
        self.assertTrue(summary['wilcoxon']['significant'])
        self.assertEqual(summary['effect_size']['rank_biserial'], 1.0)

    def test_paired_statistics_report_zero_differences_and_average_tied_ranks(self):
        analyze = self.require_helper('compute_paired_statistics')
        base = [0.1, 0.2, 0.3, 0.4, 0.5]
        augmented = [0.2, 0.3, 0.3, 0.6, 0.6]

        summary = analyze(base, augmented)

        self.assertEqual(summary['wilcoxon']['zero_difference_count'], 1)
        self.assertEqual(summary['wilcoxon']['nonzero_pair_count'], 4)
        self.assertEqual(summary['wilcoxon']['positive_rank_sum'], 10.0)
        self.assertEqual(summary['wilcoxon']['negative_rank_sum'], 0.0)
        self.assertEqual(summary['effect_size']['rank_biserial'], 1.0)

    def test_all_zero_differences_have_no_estimable_effect(self):
        analyze = self.require_helper('compute_paired_statistics')
        scores = [0.125, 0.25, 0.5, 0.75]

        summary = analyze(scores, scores)

        self.assertEqual(summary['wilcoxon']['statistic'], 0.0)
        self.assertEqual(summary['wilcoxon']['p_value'], 1.0)
        self.assertIsNone(summary['effect_size']['rank_biserial'])
        self.assertEqual(summary['wilcoxon']['zero_difference_count'], 4)

    def test_constant_differences_mark_normality_diagnostics_unavailable(self):
        analyze = self.require_helper('compute_paired_statistics')
        base = [0.125, 0.25, 0.375]
        augmented = [0.25, 0.375, 0.5]

        summary = analyze(base, augmented)

        self.assertIsNone(summary['diagnostics']['shapiro_w'])
        self.assertIsNone(summary['diagnostics']['shapiro_p_value'])
        self.assertIsNone(summary['diagnostics']['skewness'])
        self.assertIn('constant', summary['diagnostics']['unavailable_reason'])

    def test_paired_statistics_reject_different_sample_counts(self):
        analyze = self.require_helper('compute_paired_statistics')

        with self.assertRaisesRegex(ValueError, 'same number of IoU values'):
            analyze([0.1, 0.2], [0.2])

    def test_analysis_outputs_record_settings_and_create_diagnostic_plots(self):
        write_summary = self.require_helper('write_summary_json')
        write_plots = self.require_helper('write_diagnostic_plots')
        rows = [
            {
                'image_id': 'image-a',
                'iou_base': 0.3,
                'iou_augmented': 0.4,
                'iou_difference': 0.1,
                'dice_base': 0.4,
                'dice_augmented': 0.5,
                'dice_difference': 0.1,
                'ground_truth_empty': False,
            },
            {
                'image_id': 'image-b',
                'iou_base': 0.0,
                'iou_augmented': 0.0,
                'iou_difference': 0.0,
                'dice_base': 0.0,
                'dice_augmented': 0.0,
                'dice_difference': 0.0,
                'ground_truth_empty': True,
            },
        ]
        statistics = cli.compute_paired_statistics([0.3, 0.0], [0.4, 0.0])
        settings = {
            'base_checkpoint': 'base.pth',
            'augmented_checkpoint': 'augmented.pth',
            'threshold': 0.5,
            'alpha': 0.05,
            'alternative': 'greater',
            'batch_size': 2,
            'device': 'cpu',
            'test_dir': 'canonical-test',
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / 'summary.json'
            write_summary(statistics, settings, rows, summary_path)
            plot_paths = write_plots(rows, root)
            with summary_path.open(encoding='utf-8') as summary_file:
                saved_summary = __import__('json').load(summary_file)

            self.assertTrue(all(path.is_file() for path in plot_paths))

        self.assertEqual(saved_summary['effective_settings'], settings)
        self.assertTrue(saved_summary['environment']['scipy'])
        self.assertEqual(saved_summary['data_integrity']['validated_pair_count'], 2)
        self.assertEqual(saved_summary['data_integrity']['empty_ground_truth_count'], 1)
        self.assertEqual(len(saved_summary['tested_ids_sha256']), 64)
        self.assertIn('empty union => IoU 0', saved_summary['metric_convention'])
        self.assertEqual({path.name for path in plot_paths}, {
            'paired_iou.png', 'iou_difference.png'
        })

    def test_cli_defaults_use_checkpoints_and_platform_canonical_test_set(self):
        parse_args = self.require_helper('parse_args')

        args = parse_args([])

        self.assertEqual(Path(args.base_checkpoint), cli.BASE_CHECKPOINT)
        self.assertEqual(Path(args.aug_checkpoint), cli.AUGMENTED_CHECKPOINT)
        self.assertEqual(Path(args.test_dir), cli.default_test_dir())
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.alpha, 0.05)
        self.assertEqual(args.alternative, 'greater')
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.device, 'auto')

    def test_cli_accepts_path_and_statistical_overrides(self):
        parse_args = self.require_helper('parse_args')

        args = parse_args([
            '--base-checkpoint', 'base.pth',
            '--aug-checkpoint', 'augmented.pth',
            '--test-dir', 'custom-test',
            '--output-dir', 'custom-output',
            '--threshold', '0.7',
            '--alpha', '0.01',
            '--alternative', 'two-sided',
            '--batch-size', '4',
            '--device', 'cpu',
        ])

        self.assertEqual(args.base_checkpoint, 'base.pth')
        self.assertEqual(args.aug_checkpoint, 'augmented.pth')
        self.assertEqual(args.test_dir, 'custom-test')
        self.assertEqual(args.output_dir, 'custom-output')
        self.assertEqual(args.threshold, 0.7)
        self.assertEqual(args.alpha, 0.01)
        self.assertEqual(args.alternative, 'two-sided')
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.device, 'cpu')

    def test_cli_rejects_invalid_parameter_values(self):
        parse_args = self.require_helper('parse_args')
        invalid_arguments = [
            ['--threshold', '1.1'],
            ['--alpha', '0'],
            ['--alternative', 'unsupported'],
            ['--batch-size', '0'],
            ['--device', 'gpu'],
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(arguments)

    def test_validate_inputs_rejects_missing_checkpoint_and_test_paths(self):
        parse_args = self.require_helper('parse_args')
        validate_inputs = self.require_helper('validate_inputs')

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = parse_args([
                '--base-checkpoint', str(root / 'missing-base.pth'),
                '--aug-checkpoint', str(root / 'missing-augmented.pth'),
                '--test-dir', str(root / 'missing-test'),
            ])
            with self.assertRaisesRegex(FileNotFoundError, 'base checkpoint not found'):
                validate_inputs(args)

            base_checkpoint = root / 'base.pth'
            augmented_checkpoint = root / 'augmented.pth'
            base_checkpoint.touch()
            augmented_checkpoint.touch()
            args.base_checkpoint = str(base_checkpoint)
            args.aug_checkpoint = str(augmented_checkpoint)
            with self.assertRaisesRegex(FileNotFoundError, 'directory not found'):
                validate_inputs(args)


if __name__ == '__main__':
    unittest.main()
