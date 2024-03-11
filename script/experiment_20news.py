import pickle
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.getenv("PROJECT_PATH"))
from utils import get_sketcher, compute_sample_size, plot_parameters

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

def preprocess_data(categories, doc_len_filter=200):
    # Define categories and fetch data
    newsgroups_data = fetch_20newsgroups(subset='all', categories=categories)
    data, target = newsgroups_data.data, newsgroups_data.target
    # Vectorize data
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", ngram_range=(1,2))
    vectors_tfidf = vectorizer.fit_transform(data).A
    # Convert to binary matrix
    vectors_ind_tfidf = vectors_tfidf.copy()
    vectors_ind_tfidf[vectors_ind_tfidf > 0] = 1
    # Apply document length filter
    individual_words = np.sum(vectors_ind_tfidf, axis=1)
    mask = individual_words > doc_len_filter
    vectors = vectors_tfidf[mask]
    vectors_ind = vectors_ind_tfidf[mask]
    target = target[mask]
    return vectors, vectors_ind, target


def ips_topics(vectors, targets, debug=False):
    vec_num = vectors.shape[0]
    same_topic_ips = []
    diff_topic_ips = []
    for vecA_ind in range(vec_num):
        if vecA_ind%100 == 0 and debug:
            print("vecA_ind", vecA_ind)
        vecA, yA = vectors[vecA_ind], targets[vecA_ind]
        for vecB_ind in range(vecA_ind+1, vec_num, 1):
            vecB, yB = vectors[vecB_ind], targets[vecB_ind]
            ip = np.dot(vecA, vecB)
            if yA==yB:
                same_topic_ips.append(ip)
            else:
                diff_topic_ips.append(ip)
    return same_topic_ips, diff_topic_ips


def ips_est_topics(vec_num, sketch_vectors, targets, debug=False):
    same_topic_ips = []
    diff_topic_ips = []
    for vecA_ind in range(vec_num):
        if vecA_ind%100 == 0 and debug:
            print("vecA_ind", vecA_ind)
        vecA_sk, yA = sketch_vectors[vecA_ind], targets[vecA_ind]
        for vecB_ind in range(vecA_ind+1, vec_num, 1):
            vecB_sk, yB = sketch_vectors[vecB_ind], targets[vecB_ind]
            ip_est = vecA_sk.inner_product(vecB_sk)
            if yA==yB:
                same_topic_ips.append(ip_est)
            else:
                diff_topic_ips.append(ip_est)
    return same_topic_ips, diff_topic_ips

def ips_docLen(vectors, targets, debug=False):
    vec_num = vectors.shape[0]
    same_topic_ips = []
    diff_topic_ips = []
    for vecA_ind in range(vec_num):
        if vecA_ind%100 == 0 and debug:
            print("vecA_ind", vecA_ind)
        vecA, yA = vectors[vecA_ind], targets[vecA_ind]
        for vecB_ind in range(vecA_ind+1, vec_num, 1):
            vecB, yB = vectors[vecB_ind], targets[vecB_ind]
            ip = np.dot(vecA, vecB)
            if yA==yB:
                same_topic_ips.append([ip, np.sum(vecA), np.sum(vecB)])
            else:
                diff_topic_ips.append([ip, np.sum(vecA), np.sum(vecB)])
    return same_topic_ips, diff_topic_ips

def compute_diff(ips, ips_len, ips_est, len_filter=0):
    res_diff = []
    for ip, ip_len, ip_est in zip(ips, ips_len, ips_est):
        if ip_len[1]<len_filter or ip_len[2]<len_filter:
            continue
        ip_diff = ip_est-ip
        res_diff.append(abs(ip_diff))
    return res_diff

if __name__ == '__main__':
    categories = ['alt.atheism', 'comp.graphics']
    doc_len_filter = 200
    vectors, vectors_ind, target = preprocess_data(categories, doc_len_filter)

    seed = 9
    mode = "ip"
    t = 1
    vec_num = vectors.shape[0]
    sparse_vec_size = vectors.shape[1]
    sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_uniform', 'ts_2norm', 'ps_uniform', 'ps_2norm']
    storage_sizes = [i for i in range(50,301,50)]

    log_sk_name = 'log/20news_sketches'
    log_sketches = {storage_size:{sketch_method:[] for sketch_method in sketch_methods} for storage_size in storage_sizes}
    for storage_size in storage_sizes:
        for vec_ind in range(vec_num):
            if vec_ind % 20 == 0:
                print("vec_ind/vec_num", vec_ind, vec_num)
            vec = vectors[vec_ind]
            wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size = compute_sample_size(t, mode, storage_size)
            for sketch_method in sketch_methods:
                sketcher = get_sketcher(wmh_sample_size, kmv_sample_size, mh_sample_size, jl_sample_size, cs_sample_size, priority_sample_size, threshold_sample_size, sketch_method, t, seed)
                sk = sketcher.sketch(vec)
                log_sketches[storage_size][sketch_method].append(sk)
            pickle.dump(log_sketches, open(log_sk_name, 'wb'))

    log_ips_name = 'log/20news_ips'
    data_ips = {sketch_method:{storage_size:[] for storage_size in storage_sizes} for sketch_method in sketch_methods}
    ip_same_group, ip_diff_group = ips_topics(vectors, target)
    data_ips['true'] = (ip_same_group, ip_diff_group)
    for storage_size in storage_sizes:
        print(',storage_sie', storage_size)
        for sketch_method in sketch_methods:
            sketch_vectors = log_sketches[storage_size][sketch_method]
            ip_est_same_group, ip_est_diff_group = ips_est_topics(vec_num, sketch_vectors, target)
            data_ips[sketch_method][storage_size] = (ip_est_same_group, ip_est_diff_group)
            pickle.dump(data_ips, open(log_ips_name, 'wb'))

    # plot 1
    ips_len, _ = ips_docLen(vectors_ind, target)
    data_ips = pickle.load(open("log/20news_ips","rb"))
    ips, _ = data_ips['true']
    plot_data = {}
    sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_uniform', 'ps_uniform']
    storage_sizes = [i for i in range(50,301,50)]
    for sketch_method in sketch_methods:
        sub_data = defaultdict(list)
        for storage_size in storage_sizes:
            ips_est, _ = data_ips[sketch_method][storage_size]
            stats_diff = compute_diff(ips, ips_len, ips_est)
            sub_data['avg'].append(np.mean(stats_diff))
        plot_data[sketch_method] = sub_data
    
    plt.rcParams.update({'font.size': 16})
    x = storage_sizes
    for sketch_method in sketch_methods:
        y = plot_data[sketch_method]['avg']
        print("="*33)
        print(f'sketch_method: {sketch_method}\ny: {", ".join([f"{i:.6f}" for i in y])}')
        plt.plot(x, y, 
                linestyle=plot_parameters[sketch_method][3],
                label=plot_parameters[sketch_method][0], 
                marker=plot_parameters[sketch_method][1], 
                color=plot_parameters[sketch_method][2])
    plt.ylim(bottom=-0.003)
    plt.xlabel('Storage Size', weight='bold')
    plt.ylabel('Average Difference', weight='bold')
    plt.savefig('fig/20news_all.pdf', bbox_inches='tight')
    plt.close()

    # plot 2
    ips_len, _ = ips_docLen(vectors_ind, target)
    data_ips = pickle.load(open("log/20news_ips","rb"))
    ips, _ = data_ips['true']
    plot_data = {}
    sketch_methods = ['jl', 'cs', 'mh', 'wmh', 'ts_2norm', 'ps_2norm', 'ts_uniform', 'ps_uniform']
    storage_sizes = [i for i in range(50,301,50)]
    for sketch_method in sketch_methods:
        sub_data = defaultdict(list)
        for storage_size in storage_sizes:
            ips_est, _ = data_ips[sketch_method][storage_size]
            stats_diff = compute_diff(ips, ips_len, ips_est, len_filter=500)
            sub_data['avg'].append(np.mean(stats_diff))
        plot_data[sketch_method] = sub_data
        
    plt.rcParams.update({'font.size': 16})
    x = storage_sizes
    for sketch_method in sketch_methods:
        y = plot_data[sketch_method]['avg']
        print("="*33)
        print(f'sketch_method: {sketch_method}\ny: {", ".join([f"{i:.6f}" for i in y])}')
        plt.plot(x, y, 
                linestyle=plot_parameters[sketch_method][3],
                label=plot_parameters[sketch_method][0], 
                marker=plot_parameters[sketch_method][1], 
                color=plot_parameters[sketch_method][2])

    plt.ylim(bottom=-0.003)
    plt.xlabel('Storage Size', weight='bold')
    plt.ylabel('Average Difference', weight='bold')
    plt.savefig('fig/20news_500.pdf', bbox_inches='tight')
    plt.close()