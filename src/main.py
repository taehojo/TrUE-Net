import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataset import load_data, split_into_windows_as_sequence, SequenceSNPDataset
from model import TransformerClassifier
from training import train_model_mc, evaluate_model_mc
from evaluation import (calc_basic_metrics, confusion_detail, get_subgroup_metrics,
                        get_alpha_score_AUC, get_threshold_score,
                        make_plots_for_cv_alpha, make_plots_for_test)

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py [raw_file.csv] [dx_file.csv] [prefix]")
        sys.exit(1)

    raw_file = sys.argv[1]
    dx_file  = sys.argv[2]
    prefix   = sys.argv[3]

    test_size = 0.5
    window_size = 100
    epochs = 5
    mc_passes = 5
    lambda_unc = 0.05
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data, dx_data = load_data(raw_file, dx_file)
    y = dx_data["New_Label"].values
    X_3d = split_into_windows_as_sequence(raw_data, window_size)
    X_flat = X_3d.reshape(X_3d.shape[0], -1)

    X_3d_train, X_3d_test, X_flat_train, X_flat_test, y_train, y_test = \
        train_test_split(X_3d, X_flat, y, stratify=y, test_size=test_size, random_state=42)

    print(f"[INFO] total {len(y)} samples: {len(y_train)}(Train), {len(y_test)}(Test)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_train = len(y_train)

    oof_probs_t = np.zeros(n_train, dtype=np.float64)
    oof_vars_t  = np.zeros(n_train, dtype=np.float64)
    oof_probs_rf= np.zeros(n_train, dtype=np.float64)
    oof_vars_rf = np.zeros(n_train, dtype=np.float64)
    oof_labels  = np.array(y_train, copy=True)

    alpha_candidates = np.linspace(0, 1, 6)
    alpha_scores_list = []
    fold_id = 0

    for tr_idx, val_idx in skf.split(X_3d_train, y_train):
        fold_id += 1
        print(f"\n[Fold {fold_id}] train={len(tr_idx)} val={len(val_idx)}")

        X_fold_tr_3d = X_3d_train[tr_idx]
        X_fold_val_3d= X_3d_train[val_idx]
        y_fold_tr = y_train[tr_idx]
        y_fold_val= y_train[val_idx]

        X_fold_tr_flat= X_flat_train[tr_idx]
        X_fold_val_flat= X_flat_train[val_idx]

        ds_fold_tr  = SequenceSNPDataset(X_fold_tr_3d, y_fold_tr)
        ds_fold_val = SequenceSNPDataset(X_fold_val_3d, y_fold_val)

        loader_tr   = torch.utils.data.DataLoader(ds_fold_tr, batch_size=32, shuffle=True)
        loader_val  = torch.utils.data.DataLoader(ds_fold_val, batch_size=32, shuffle=False)

        model_t = TransformerClassifier(input_dim=window_size, dropout=0.2).to(device)
        opt_t = torch.optim.AdamW(model_t.parameters(), lr=1e-3)
        train_model_mc(model_t, loader_tr, opt_t, device, mc_passes=mc_passes, lambda_unc=lambda_unc, epochs=epochs)

        labels_val_t, preds_val_t, probs_val_t, vars_val_t = evaluate_model_mc(model_t, loader_val, device, mc_passes=mc_passes)
        probs_val_t = np.array(probs_val_t, dtype=np.float64)
        vars_val_t  = np.array(vars_val_t,  dtype=np.float64)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_fold_tr_flat, y_fold_tr)
        rf_probs_val = rf.predict_proba(X_fold_val_flat)[:,1].astype(np.float64)
        tree_probas  = np.array([tree.predict_proba(X_fold_val_flat)[:,1] for tree in rf.estimators_])
        rf_vars_val  = np.var(tree_probas, axis=0).astype(np.float64)

        oof_probs_t[val_idx] = probs_val_t
        oof_vars_t[val_idx]  = vars_val_t
        oof_probs_rf[val_idx]= rf_probs_val
        oof_vars_rf[val_idx] = rf_vars_val

        row_score = {"fold":fold_id}
        arr_labels_val = np.array(labels_val_t)
        for a in alpha_candidates:
            p_tmp = a*probs_val_t + (1-a)*rf_probs_val
            if len(np.unique(arr_labels_val))<2:
                auc_ = 0.5
            else:
                auc_ = roc_auc_score(arr_labels_val, p_tmp)
            row_score[f"a={a:.1f}"] = auc_
        alpha_scores_list.append(row_score)

    df_cv_alpha = pd.DataFrame(alpha_scores_list)
    os.makedirs("result", exist_ok=True)
    cv_alpha_csv = f"result/{prefix}_cv_alpha_scores.csv"
    df_cv_alpha.to_csv(cv_alpha_csv, index=False)
    print(f"[INFO] CV alpha scores per fold CSV -> {cv_alpha_csv}")

    make_plots_for_cv_alpha(df_cv_alpha, prefix=prefix)
    print("[INFO] OOF completed.")

    arr_labels = oof_labels
    best_alpha = 0.0
    best_alpha_score = -999.0
    alpha_score_map = {}
    for a in alpha_candidates:
        p_ens_oof = a*oof_probs_t + (1-a)*oof_probs_rf
        if len(np.unique(arr_labels))<2:
            score_ = 0.5
        else:
            score_ = roc_auc_score(arr_labels, p_ens_oof)
        alpha_score_map[a] = score_
        if score_>best_alpha_score:
            best_alpha_score=score_
            best_alpha=a

    df_alpha_map = pd.DataFrame([{"alpha":k,"AUC":v} for k,v in alpha_score_map.items()])
    df_alpha_map.to_csv(f"result/{prefix}_oof_alpha_AUC_map.csv", index=False)
    print(f"[INFO] best alpha by AUC: alpha={best_alpha}, AUC={best_alpha_score:.4f}")

    p_ens_oof = best_alpha*oof_probs_t + (1-best_alpha)*oof_probs_rf
    v_ens_oof = best_alpha*oof_vars_t  + (1-best_alpha)*oof_vars_rf

    var_min, var_max = v_ens_oof.min(), v_ens_oof.max()
    thr_candidates = np.linspace(var_min, var_max, 10)
    best_thr, best_thr_score = var_min, -999.0
    thr_score_map={}
    for thr_ in thr_candidates:
        s_ = get_threshold_score(arr_labels, p_ens_oof, v_ens_oof, thr_)
        thr_score_map[thr_] = s_
        if s_>best_thr_score:
            best_thr_score=s_
            best_thr=thr_

    df_thr_map = pd.DataFrame([{"threshold":k,"uncer+cer_auc":v} for k,v in thr_score_map.items()])
    df_thr_map.to_csv(f"result/{prefix}_oof_threshold_score.csv", index=False)
    print(f"[INFO] best threshold={best_thr:.4f}, score={best_thr_score:.4f}")

    ds_train_all = SequenceSNPDataset(X_3d_train, y_train)
    loader_train_all = torch.utils.data.DataLoader(ds_train_all, batch_size=32, shuffle=True)
    final_model_t= TransformerClassifier(input_dim=window_size, dropout=0.2).to(device)
    opt_final_t  = torch.optim.AdamW(final_model_t.parameters(), lr=1e-3)
    train_model_mc(final_model_t, loader_train_all, opt_final_t, device, mc_passes=mc_passes, lambda_unc=lambda_unc, epochs=epochs)

    rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_final.fit(X_flat_train, y_train)

    ds_test = SequenceSNPDataset(X_3d_test, y_test)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False)
    test_labels_t, _, test_probs_t, test_vars_t = evaluate_model_mc(final_model_t, loader_test, device, mc_passes=mc_passes)
    test_probs_rf= rf_final.predict_proba(X_flat_test)[:,1]
    tree_probas_test = np.array([tree.predict_proba(X_flat_test)[:,1] for tree in rf_final.estimators_])
    test_vars_rf = np.var(tree_probas_test, axis=0)

    test_probs_t= np.array(test_probs_t,dtype=np.float64)
    test_probs_rf= np.array(test_probs_rf,dtype=np.float64)
    test_vars_t= np.array(test_vars_t,dtype=np.float64)
    test_vars_rf= np.array(test_vars_rf,dtype=np.float64)

    test_ens_prob= best_alpha*test_probs_t + (1-best_alpha)*test_probs_rf
    test_ens_var = best_alpha*test_vars_t  + (1-best_alpha)*test_vars_rf

    group_list=[]
    for v in test_ens_var:
        if v>=best_thr:
            group_list.append("Uncertain")
        else:
            group_list.append("Certain")

    test_preds_ens= (test_ens_prob>=0.5).astype(int)
    n_total= len(y_test)
    all_m = calc_basic_metrics(y_test, test_preds_ens, test_ens_prob)
    all_cm= confusion_detail(y_test, test_preds_ens)

    unc_idx= [i for i,v in enumerate(group_list) if v=="Uncertain"]
    cer_idx= [i for i,v in enumerate(group_list) if v=="Certain"]
    n_unc= len(unc_idx)
    n_cer= len(cer_idx)
    pct_unc= (100.0*n_unc/n_total) if n_total else 0
    pct_cer= (100.0*n_cer/n_total) if n_total else 0

    y_unc= y_test[unc_idx]
    p_unc= test_ens_prob[unc_idx]
    prd_unc= test_preds_ens[unc_idx]
    unc_m= calc_basic_metrics(y_unc, prd_unc, p_unc)
    unc_cm= confusion_detail(y_unc, prd_unc)

    y_cer= y_test[cer_idx]
    p_cer= test_ens_prob[cer_idx]
    prd_cer= test_preds_ens[cer_idx]
    cer_m= calc_basic_metrics(y_cer, prd_cer, p_cer)
    cer_cm= confusion_detail(y_cer, prd_cer)

    print("\n=== [TEST RESULT] ===")
    print(f"All (#={n_total}): {all_m}, CM={all_cm}")
    print(f"Uncertain (#={n_unc}, {pct_unc:.1f}%): {unc_m}, CM={unc_cm}")
    print(f"Certain   (#={n_cer}, {pct_cer:.1f}%): {cer_m}, CM={cer_cm}")

    rows_test=[]
    for i in range(n_total):
        rows_test.append({
            "sample_idx": i,
            "true_label": int(y_test[i]),
            "final_prob": test_ens_prob[i],
            "final_var": test_ens_var[i],
            "group_uncertain_or_certain": group_list[i]
        })

    df_details= pd.DataFrame(rows_test)
    os.makedirs("result", exist_ok=True)
    detail_csv= f"result/{prefix}_test_details.csv"
    df_details.to_csv(detail_csv, index=False, encoding='utf-8-sig')

    sum_rows=[]
    def add_sum(name_,n_,pct_,m_,cm_):
        r_= {"Group":name_,"N":n_,"Pct":pct_}
        r_.update(m_)
        r_.update(cm_)
        return r_

    s_all= add_sum("All",n_total,100.0,all_m,all_cm)
    s_unc= add_sum("Uncertain",n_unc,pct_unc,unc_m,unc_cm)
    s_cer= add_sum("Certain",n_cer,pct_cer,cer_m,cer_cm)
    df_sum= pd.DataFrame([s_all,s_unc,s_cer])
    sum_csv= f"result/{prefix}_test_summary.csv"
    df_sum.to_csv(sum_csv, index=False, encoding='utf-8-sig')

    print(f"[INFO] test details -> {detail_csv}")
    print(f"[INFO] test summary -> {sum_csv}")
    make_plots_for_test(df_details, prefix=prefix)

if __name__=="__main__":
    main()
