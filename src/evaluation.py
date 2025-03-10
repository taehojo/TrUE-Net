import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, roc_auc_score

def calc_basic_metrics(labels, preds, probs):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    labels = np.array(labels)
    preds  = np.array(preds)
    probs  = np.array(probs)
    if len(labels) == 0:
        return {'ACC':0,'AUC':0,'F1':0,'Sensitivity':0,'Specificity':0,'Precision':0,'Recall':0}
    unique_labels = np.unique(labels)
    if len(unique_labels)==1:
        acc = accuracy_score(labels, preds)
        auc_ = 0.5
        f1_ = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec  = recall_score(labels, preds, zero_division=0)
        return {'ACC':acc,'AUC':auc_,'F1':f1_,'Sensitivity':rec,'Specificity':0,'Precision':prec,'Recall':rec}
    acc = accuracy_score(labels, preds)
    auc_ = roc_auc_score(labels, probs)
    f1_ = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp/(tp+fn) if (tp+fn) else 0
    specificity = tn/(tn+fp) if (tn+fp) else 0
    return {
        'ACC':acc,'AUC':auc_,'F1':f1_,
        'Sensitivity':sensitivity,'Specificity':specificity,
        'Precision':prec,'Recall':rec
    }

def confusion_detail(labels, preds):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    if len(labels) == 0 or len(preds) == 0:
        return {"TN":0, "FP":0, "FN":0, "TP":0}
    
    unique_labels = np.unique(labels)
    unique_preds = np.unique(preds)
    
    if len(unique_labels) < 2 or len(unique_preds) < 2:
        if len(unique_labels) == 1 and len(unique_preds) == 1:
            if unique_labels[0] == 1 and unique_preds[0] == 1:
                return {"TN":0, "FP":0, "FN":0, "TP":len(labels)}
            elif unique_labels[0] == 0 and unique_preds[0] == 0:
                return {"TN":len(labels), "FP":0, "FN":0, "TP":0}
            else:
                if unique_labels[0] == 1:
                    return {"TN":0, "FP":0, "FN":len(labels), "TP":0}
                else:
                    return {"TN":0, "FP":len(labels), "FN":0, "TP":0}
        return {"TN":0, "FP":0, "FN":0, "TP":0}
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {"TN":tn, "FP":fp, "FN":fn, "TP":tp}

def get_subgroup_metrics(labels, probs, var_, threshold):
    labels = np.array(labels)
    probs = np.array(probs)
    var_  = np.array(var_)
    unc_mask = (var_ >= threshold)
    cer_mask = (var_ < threshold)
    labels_unc = labels[unc_mask]
    probs_unc  = probs[unc_mask]
    preds_unc  = (probs_unc>=0.5).astype(int)
    unc_m = calc_basic_metrics(labels_unc, preds_unc, probs_unc)
    labels_cer = labels[cer_mask]
    probs_cer  = probs[cer_mask]
    preds_cer  = (probs_cer>=0.5).astype(int)
    cer_m = calc_basic_metrics(labels_cer, preds_cer, probs_cer)
    return unc_m, cer_m

def get_alpha_score_AUC(arr_labels, p_ens):
    arr_labels = np.array(arr_labels)
    p_ens = np.array(p_ens)
    if len(np.unique(arr_labels))<2:
        return 0.5
    return roc_auc_score(arr_labels, p_ens)

def get_threshold_score(arr_labels, p_ens, v_ens, thr):
    unc_m, cer_m = get_subgroup_metrics(arr_labels, p_ens, v_ens, thr)
    return unc_m["AUC"] + cer_m["AUC"]

def make_plots_for_test(df_details, prefix="myprefix"):
    os.makedirs("result", exist_ok=True)
    pdf_filename = f"result/{prefix}_analysis_plots.pdf"

    sns.set_theme(style="whitegrid")
    df_all = df_details
    df_unc = df_all[df_all["group_uncertain_or_certain"]=="Uncertain"]
    df_cer = df_all[df_all["group_uncertain_or_certain"]=="Certain"]

    y_all = df_all["true_label"].values
    p_all = df_all["final_prob"].values

    y_unc = df_unc["true_label"].values
    p_unc = df_unc["final_prob"].values

    y_cer = df_cer["true_label"].values
    p_cer = df_cer["final_prob"].values

    def get_metrics(y_true, p_prob):
        prd = (p_prob>=0.5).astype(int)
        return calc_basic_metrics(y_true, prd, p_prob)

    m_all = get_metrics(y_all, p_all)
    m_unc = get_metrics(y_unc, p_unc)
    m_cer = get_metrics(y_cer, p_cer)

    fig_list = []

    fig_kde, ax_kde = plt.subplots(figsize=(6,6))
    color_unc = "#b3cde3"
    color_cer = "#005b96"
    if len(df_unc)>0:
        sns.kdeplot(x="final_prob", data=df_unc, fill=True, color=color_unc, label="Uncertain", ax=ax_kde)
    if len(df_cer)>0:
        sns.kdeplot(x="final_prob", data=df_cer, fill=True, color=color_cer, label="Certain", ax=ax_kde)
    ax_kde.set_title("final_prob KDE (Uncertain vs Certain)", fontsize=12, fontweight="bold")
    ax_kde.set_xlabel("final_prob")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()
    ax_kde.grid(alpha=0.4)
    fig_list.append(fig_kde)

    fig_bar, ax_bar = plt.subplots(figsize=(6,6))
    metric_names = ["ACC","AUC","F1"]
    group_labels = ["Uncertain","All","Certain"]
    group_colors = [color_unc, "#6497b1", color_cer]
    data_unc = [m_unc[m] for m in metric_names]
    data_all = [m_all[m] for m in metric_names]
    data_cer = [m_cer[m] for m in metric_names]
    data_for_bar = [data_unc, data_all, data_cer]
    x_pos = np.arange(len(metric_names))
    bar_width = 0.25
    for i, group in enumerate(group_labels):
        ax_bar.bar(x_pos + i*bar_width, data_for_bar[i], width=bar_width,
                   color=group_colors[i], alpha=0.8, label=group)
    ax_bar.set_xticks(x_pos + bar_width)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylim(0,1.1)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("ACC, AUC, F1 by group", fontsize=12, fontweight="bold")
    ax_bar.legend()
    ax_bar.grid(alpha=0.4)
    fig_list.append(fig_bar)

    fig_scatter, ax_scatter = plt.subplots(figsize=(6,6))
    df_unc_scatter = df_unc.copy()
    df_cer_scatter = df_cer.copy()
    ax_scatter.scatter(df_unc_scatter["final_prob"], df_unc_scatter["final_var"],
                       color=color_unc, alpha=0.7, label="Uncertain")
    ax_scatter.scatter(df_cer_scatter["final_prob"], df_cer_scatter["final_var"],
                       color=color_cer, alpha=0.7, label="Certain")
    ax_scatter.set_title("Scatter: final_prob vs final_var", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("final_prob")
    ax_scatter.set_ylabel("final_var")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.4)
    fig_list.append(fig_scatter)

    with PdfPages(pdf_filename) as pdf:
        for f in fig_list:
            pdf.savefig(f)

    plt.close('all')
    print(f"pdf created.")