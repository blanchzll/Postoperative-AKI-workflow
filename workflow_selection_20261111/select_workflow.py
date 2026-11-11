from pathlib import Path
import argparse
import csv
import json
import io
import zipfile
from decimal import Decimal

parser = argparse.ArgumentParser(description='Query the existing resource-planning grid.')
parser.add_argument('--evaluation', choices=['geography', 'temporal'], required=True)
parser.add_argument('--budget', choices=['10', '20'], default='20')
parser.add_argument('--cost27', required=True)
parser.add_argument('--cost38', required=True)
parser.add_argument('--grid', type=Path, default=Path(__file__).parent/'WORKFLOW_CHOICE_AND_SHAP_CODE.zip')
args = parser.parse_args()
a27 = Decimal(args.cost27)
a38 = Decimal(args.cost38)
if not (0 <= a27 <= a38 <= Decimal('.30')):
    parser.error('The evaluated nested cost grid requires 0 <= cost27 <= cost38 <= .30.')
if a27*100 != int(a27*100) or a38*100 != int(a38*100):
    parser.error('The evaluated grid uses cost-ratio steps of .01.')
if args.grid.suffix == '.zip':
    archive = zipfile.ZipFile(args.grid)
    source = io.TextIOWrapper(archive.open('shap_threshold_reporting_20261111_v4/three_workflow_choice/STRATEGY_SELECTION_GRID.csv'), encoding='utf-8')
else:
    archive = None
    source = args.grid.open()
with source:
    matches = [
        row for row in csv.DictReader(source)
        if row['line'] == args.evaluation
        and row['mode'] == 'calibrated'
        and row['budget_percent'] == args.budget
        and Decimal(row['cost27']) == a27
        and Decimal(row['cost38']) == a38
    ]
if archive is not None:
    archive.close()
if len(matches) != 1:
    raise ValueError('Expected one matching evaluated scenario.')
print(json.dumps(matches[0], indent=2))
