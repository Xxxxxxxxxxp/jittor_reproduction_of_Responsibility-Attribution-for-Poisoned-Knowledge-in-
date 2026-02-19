

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score



def dynamic_threshold(re_score,true_label):
    """      List[List[float]]   List[List[float]]"""
    y_true=[]
    """真实标签"""
    y_predict=[]
    """预测"""
    tn_list, fp_list, fn_list, tp_list=[], [], [], []
    """四种判断结果0	0	TN	本来没投毒 → 也预测没投毒
                  0	1	FP	本来没投毒 → 被冤枉成投毒
                  1	0	FN	本来是投毒 → 被漏掉了
                  1	1	TP              
                    list[int]                     """
    for idx in range(len(re_score)):
        iter_score=re_score[idx]
        iter_label=true_label[idx]

        kmeans=KMeans(n_clusters=2, random_state=42)
        probablity=np.array(iter_score).reshape(-1,1)
        kmeans.fit(probablity)
        centers=kmeans.cluster_centers_.flatten()
        label_for_larger_center=np.argmax(centers)
        labels=kmeans.labels_
        y_predict_sub=np.zeros_like(kmeans.labels_)
        y_predict_sub[labels == label_for_larger_center]=1

        y_predict.extend(y_predict_sub.tolist())
        y_true.extend(iter_label)

        tn, fp, fn, tp=confusion_matrix(iter_label, y_predict_sub.tolist()).ravel()
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
    accuracy=accuracy_score(y_true, y_predict)
    tn, fp, fn, tp=confusion_matrix(y_true, y_predict).ravel()
    fpr=fp / (fp + tn)  
    fnr=fn / (fn + tp) 

    return tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_predict





def evaluate_result(y_true, trace_scores_dict, variant):
    tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred=dynamic_threshold(trace_scores_dict[f"variant_{variant}"], y_true)

    return tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred






