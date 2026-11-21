from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
import unittest

from check_sources import load_sources

DATA = Path(__file__).with_name('data')


class ExchangeTests(unittest.TestCase):
    def test_example_is_readable(self):
        tables = load_sources(DATA)
        self.assertTrue(all(tables.values()))
        self.assertTrue(any(r['repair_of'] for r in tables['scale']))
        self.assertTrue(any(r['state'] == 'sent' for r in tables['links']))
        self.assertTrue(all(float(r['weight_min']) > 0 and float(r['weight_min']) <= float(r['weight_max']) and int(r['from']) <= int(r['to']) for r in tables['scale']))
        self.assertTrue(any(r['occupied_by'] for r in tables['isolation']))

    def test_duplicate_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / 'copies'
            shutil.copytree(DATA, duplicate)
            path = duplicate / 'source_batches.csv'
            rows = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(rows + [rows[1]]) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '标识为空或重复'):
                load_sources(duplicate)
