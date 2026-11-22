"""Evaluate locked AKI rankings at equal capacity and equal recorded AKI yield."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd

from fixed_prediction_metrics import _design, group_weights

CAPACITIES=np.arange(5,31,dtype=np.int64)
ROUTES=('original_nine','original_full27','original_endop38')
CONTRASTS=((ROUTES[1],ROUTES[0]),(ROUTES[2],ROUTES[1]),(ROUTES[2],ROUTES[0]))
SEED=20261017
BOOTSTRAPS=2000


def counts_at_capacities(sorted_outcomes, counts):
    return np.cumsum(sorted_outcomes)[counts-1]


def minimum_count(sorted_outcomes, target):
    if target==0:
        return 0
    return int(np.flatnonzero(sorted_outcomes)[target-1]+1)


def evaluate(data):
    columns=['operation_id','patient_group','hospital','recorded_aki',*ROUTES]
    if not set(columns).issubset(data):
        raise ValueError(f'Missing columns: {sorted(set(columns)-set(data))}')
    if data.empty or data[columns].isna().any().any() or not data.operation_id.is_unique:
        raise ValueError('Provide nonmissing data, with one row per common evaluation operation')
    if not data.recorded_aki.isin([0,1]).all():
        raise ValueError('recorded_aki must use 0 or 1')
    if not all(data[name].between(0,1).all() for name in ROUTES):
        raise ValueError('All predictions must be risks in [0,1]')
    y, _, _, _, strata, systems=_design(
        data.recorded_aki.astype(np.int64),data.patient_group,data.operation_id,data.hospital,
        {name:data[name].to_numpy() for name in ROUTES})
    size=len(y)
    n_selected=(CAPACITIES*size+99)//100
    index_20=int(np.flatnonzero(CAPACITIES==20)[0])
    ordered={model:y[system.order] for model,system in systems.items()}
    events={model:counts_at_capacities(v,n_selected) for model,v in ordered.items()}
    yield_draws={pair:np.empty((BOOTSTRAPS,len(CAPACITIES))) for pair in CONTRASTS}
    volume_draws={pair:np.empty(BOOTSTRAPS) for pair in CONTRASTS}
    rng=np.random.default_rng(SEED)
    for draw in range(BOOTSTRAPS):
        w=group_weights(strata,size,rng)
        draw_size=w.sum()
        slots=(CAPACITIES*draw_size+99)//100
        outcomes={name:np.repeat(ordered[name],w[system.order]) for name,system in systems.items()}
        counts={name:counts_at_capacities(v,slots) for name,v in outcomes.items()}
        for alt,ref in CONTRASTS:
            pair=(alt,ref)
            yield_draws[pair][draw]=(counts[alt]-counts[ref])*1000.0/draw_size
            target=counts[ref][index_20]
            volume_draws[pair][draw]=(minimum_count(outcomes[alt],target)-slots[index_20])*1000.0/draw_size
    grid=[]
    workload=[]
    for alt,ref in CONTRASTS:
        pair=(alt,ref)
        lower,upper=np.quantile(yield_draws[pair],[.025,.975],axis=0)
        observed=(events[alt]-events[ref])*1000.0/size
        for i,q in enumerate(CAPACITIES):
            grid.append(dict(model=alt,reference=ref,review_pct=int(q),eligible_operations=size,
                             operations_selected_per_model=int(n_selected[i]),model_AKI_selected=int(events[alt][i]),
                             reference_AKI_selected=int(events[ref][i]),difference_per_1000=float(observed[i]),
                             lower_95_pointwise=float(lower[i]),upper_95_pointwise=float(upper[i])))
        target=int(events[ref][index_20])
        alt_slots=minimum_count(ordered[alt],target)
        ref_slots=int(n_selected[index_20])
        lower,upper=np.quantile(volume_draws[pair],[.025,.975])
        workload.append(dict(model=alt,reference=ref,eligible_operations=size,reference_capacity_pct=20,
                             target_AKI_selected=target,reference_records_assessed=ref_slots,
                             model_records_assessed_at_equal_AKI=alt_slots,review_difference_records=alt_slots-ref_slots,
                             review_difference_per_1000=(alt_slots-ref_slots)*1000.0/size,
                             lower_95_pointwise=float(lower),upper_95_pointwise=float(upper)))
    return grid,workload


def write_table(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path,help='Aligned CSV for one evaluation cohort')
    parser.add_argument('--output',required=True,type=Path,help='New result directory')
    args=parser.parse_args()
    data=pd.read_csv(args.input,dtype={'operation_id':str,'patient_group':str,'hospital':str})
    grid,workload=evaluate(data)
    args.output.mkdir(parents=True,exist_ok=False)
    write_table(args.output/'CAPACITY_GRID.csv',grid)
    write_table(args.output/'EQUAL_CAPTURE_WORKLOAD.csv',workload)


if __name__=='__main__':
    main()
