import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)

def calc_basic_metrics(labels, preds, probs):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)
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
    return {'ACC':acc,'AUC':auc_,'F1':f1_,'Sensitivity':sensitivity,'Specificity':specificity,'Precision':prec,'Recall':rec}

def confusion_detail(labels, preds):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {"TN":tn,"FP":fp,"FN":fn,"TP":tp}

def get_subgroup_metrics(labels, probs, var_, threshold):
    labels = np.array(labels)
    probs = np.array(probs)
    var_ = np.array(var_)
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
    from sklearn.metrics import roc_auc_score
    arr_labels = np.array(arr_labels)
    p_ens = np.array(p_ens)
    if len(np.unique(arr_labels))<2:
        return 0.5
    return roc_auc_score(arr_labels, p_ens)

def get_threshold_score(arr_labels, p_ens, v_ens, thr):
    unc_m, cer_m = get_subgroup_metrics(arr_labels, p_ens, v_ens, thr)
    return unc_m["AUC"] + cer_m["AUC"]

def make_plots_for_cv_alpha(df_alpha, prefix="myprefix"):
    from copy import deepcopy
    os.makedirs("result", exist_ok=True)
    pdf_fn = f"result/{prefix}_cv_alpha_AUC_plot.pdf"
    alpha_cols = [c for c in df_alpha.columns if c.startswith("a=")]
    df_melt = df_alpha.melt(id_vars=["fold"], value_vars=alpha_cols, var_name="alpha_str", value_name="AUC")
    df_melt["alpha"] = df_melt["alpha_str"].apply(lambda x: float(x.split('=')[1]))
    fig, ax = plt.subplots(figsize=(6,5))
    sns.lineplot(data=df_melt, x="alpha", y="AUC", hue="fold", marker="o", ax=ax)
    ax.set_title("CV Fold-wise alpha vs AUC", fontsize=12, fontweight="bold")
    ax.set_xlabel("alpha")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.4)
    with PdfPages(pdf_fn) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"[INFO] CV fold alpha vs AUC plot -> {pdf_fn}")

def make_plots_for_test(df_details, prefix="myprefix"):
    from copy import deepcopy
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    os.makedirs("result", exist_ok=True)
    pdf_filename = f"result/{prefix}_analysis_plots.pdf"
    sns.set_theme(style="whitegrid")
    df_all = df_details
    df_unc = df_all[df_all["group_uncertain_or_certain"]=="Uncertain"]
    df_cer = df_all[df_all["group_uncertain_or_certain"]=="Certain"]
    y_all = df_all["true_label"].values
    p_all = df_all["final_prob"].values
    pred_all = (p_all>=0.5).astype(int)
    y_unc = df_unc["true_label"].values
    p_unc = df_unc["final_prob"].values
    pred_unc = (p_unc>=0.5).astype(int)
    y_cer = df_cer["true_label"].values
    p_cer = df_cer["final_prob"].values
    pred_cer = (p_cer>=0.5).astype(int)
    def plot_roc(ax, y_true, y_prob, label_txt, color_):
        if len(np.unique(y_true))<2:
            return
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_ = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{label_txt} (AUC={auc_:.3f})", color=color_, lw=2)
    def plot_pr(ax, y_true, y_prob, label_txt, color_):
        if len(np.unique(y_true))<2:
            return
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap_ = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{label_txt} (AP={ap_:.3f})", color=color_, lw=2)
    color_unc = "#b3cde3"
    color_all = "#6497b1"
    color_cer = "#005b96"
    fig_roc, ax_roc = plt.subplots(figsize=(6,6))
    plot_roc(ax_roc, y_unc, p_unc, "Uncertain", color_unc)
    plot_roc(ax_roc, y_all, p_all, "All", color_all)
    plot_roc(ax_roc, y_cer, p_cer, "Certain", color_cer)
    ax_roc.plot([0,1],[0,1],"--",color="gray", label="Random")
    ax_roc.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.4)
    fig_pr, ax_pr = plt.subplots(figsize=(6,6))
    plot_pr(ax_pr, y_unc, p_unc, "Uncertain", color_unc)
    plot_pr(ax_pr, y_all, p_all, "All", color_all)
    plot_pr(ax_pr, y_cer, p_cer, "Certain", color_cer)
    ax_pr.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.4)
    fig_kde, ax_kde = plt.subplots(figsize=(6,6))
    if len(df_unc)>0:
        sns.kdeplot(x="final_prob", data=df_unc, fill=True, color=color_unc, label="Uncertain", ax=ax_kde)
    if len(df_cer)>0:
        sns.kdeplot(x="final_prob", data=df_cer, fill=True, color=color_cer, label="Certain", ax=ax_kde)
    ax_kde.set_title("final_prob KDE (Uncertain vs Certain)", fontsize=12, fontweight="bold")
    ax_kde.set_xlabel("final_prob")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()
    ax_kde.grid(alpha=0.4)
    from .evaluation import calc_basic_metrics
    def get_metrics(y_true, p_prob):
        prd = (p_prob>=0.5).astype(int)
        return calc_basic_metrics(y_true, prd, p_prob)
    m_all = get_metrics(y_all, p_all)
    m_unc = get_metrics(y_unc, p_unc)
    m_cer = get_metrics(y_cer, p_cer)
    fig_bar, ax_bar = plt.subplots(figsize=(6,6))
    metric_names = ["ACC","AUC","F1"]
    group_labels = ["Uncertain","All","Certain"]
    group_colors = [color_unc, color_all, color_cer]
    data_unc = [m_unc[m] for m in metric_names]
    data_all = [m_all[m] for m in metric_names]
    data_cer = [m_cer[m] for m in metric_names]
    data_for_bar = [data_unc, data_all, data_cer]
    x_pos = np.arange(len(metric_names))
    bar_width = 0.25
    for i, group in enumerate(group_labels):
        ax_bar.bar(x_pos + i*bar_width, data_for_bar[i], width=bar_width, color=group_colors[i], alpha=0.8, label=group)
    ax_bar.set_xticks(x_pos + bar_width)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylim(0,1.1)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("ACC, AUC, F1 by group", fontsize=12, fontweight="bold")
    ax_bar.legend()
    ax_bar.grid(alpha=0.4)
    fig_scatter, ax_scatter = plt.subplots(figsize=(6,6))
    df_unc_scatter = df_unc.copy()
    df_cer_scatter = df_cer.copy()
    ax_scatter.scatter(df_unc_scatter["final_prob"], df_unc_scatter["final_var"], color=color_unc, alpha=0.7, label="Uncertain")
    ax_scatter.scatter(df_cer_scatter["final_prob"], df_cer_scatter["final_var"], color=color_cer, alpha=0.7, label="Certain")
    ax_scatter.set_title("Scatter: final_prob vs final_var", fontsize=12, fontweight="bold")
    ax_scatter.set_xlabel("final_prob")
    ax_scatter.set_ylabel("final_var")
    ax_scatter.legend()
    ax_scatter.grid(alpha=0.4)
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig_roc)
        pdf.savefig(fig_pr)
        pdf.savefig(fig_kde)
        pdf.savefig(fig_bar)
        pdf.savefig(fig_scatter)
    plt.close('all')
    print(f"*** All final test plots saved to '{pdf_filename}' ***")
