#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jittor as jt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[24]:


def dynamic_threshold(y_scores_2D, y_true_2D):
    """y_scores_2D  [[float，float...]]   传入的每个问题的每个文本的责任分数"""
    """y_true_2D  [[int,int...]]  传入的每个问题对应每个文本的正确标签"""
    y_pred,y_true=[],[]
    acc,tp,tn,fp,fn=[],[],[],[],[]
    for i in range(len(y_scores_2D)):
        iter_true,iter_score=y_true_2D[i],y_scores_2D[i]


        km=KMeans(n_clusters=2, random_state=42)
        np_iter_score=np.array(iter_score).reshape(-1,1)
        km.fit(np_iter_score)
        center=km.cluster_centers_.flatten()
        """更大值簇的位置标号0/1"""
        label_for_larger_cluster=np.argmax(center)

        klabel=km.labels_
        y_pred_iter=np.zeros_like(klabel)
        y_pred_iter[km.labels_==label_for_larger_cluster]=1
        y_pred.append(y_pred_iter.tolist())
        y_true.append(iter_true)


        iter_tn,iter_fp,iter_fn,iter_tp=confusion_matrix(iter_true,y_pred_iter.tolist()).ravel()
        tp.append(iter_tp)
        tn.append(iter_tn)
        fn.append(iter_fn)
        fp.append(iter_fp)

        iter_acc=accuracy_score(iter_true,y_pred_iter)
        acc.append(iter_acc)

    y_pred_faltten=np.concatenate(y_pred).tolist()
    y_true_faltten=np.concatenate(y_true).tolist()
    acc_final=accuracy_score(y_true_faltten,y_pred_faltten)
    all_tn,all_fp,all_fn,all_tp=confusion_matrix(y_true_faltten,y_pred_faltten).ravel()
    """假阳率"""
    fpr=all_fp/(all_fp+all_tn)
    """假阴率"""
    fnr=all_fn/(all_fn+all_tp)

    return tn,fp,fn,tp,fpr,fnr,acc_final,acc,y_pred


# In[31]:


def evaluate_results(y_true,trace_scores_dict,variant):
    tn_list, fp_list, fn_list, tp_list, fpr, fnr, acc_fianl,acc_list, y_pred=dynamic_threshold(trace_scores_dict[f"variant_{variant}"], y_true)
    return tn_list, fp_list, fn_list, tp_list, fpr, fnr, acc_fianl,acc_list, y_pred


# In[32]:


