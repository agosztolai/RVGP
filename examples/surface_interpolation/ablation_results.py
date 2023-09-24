#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def process_results(ablation_results, trials):
    results = []
    for i in range(len(ablation_results)):
        gt, pred = ablation_results[i]
        
        norm = np.linalg.norm(ablation_results[i][1], axis=1, keepdims=True)
        alignment = (ablation_results[i][0]*ablation_results[i][1]/norm).sum(1)
        
        alignment_mean = alignment.mean()
        alignment_min, alignment_max = st.norm.interval(confidence=0.99, 
                         loc=np.mean(alignment),
                         scale=st.sem(alignment))
        
        trial = i%trials
        if trial == 0:
            results_condition = np.zeros([trials, 3])
            
        results_condition[trial] = np.array([alignment_mean, alignment_min, alignment_max])
            
        if trial == trials - 1:
            results.append(results_condition)
            
    return results

trials = 10

ablation_density = pickle.load(open('ablation_density_results.pkl', 'rb'))
ablation_density_results = process_results(ablation_density, trials) 
density = np.linspace(0.015,0.08,10)

ablation_eigenvectors = pickle.load(open('ablation_eigenvector_results.pkl', 'rb'))
ablation_eigenvectors_results = process_results(ablation_eigenvectors, trials) 
k = [1,2,5,10,100,200,300]

fig = plt.figure()
ablation_eigenvectors_results = [r.mean(0) for r in ablation_eigenvectors_results]
ablation_eigenvectors_results = np.array(ablation_eigenvectors_results)
plt.errorbar(k, ablation_eigenvectors_results[:,0], yerr = ablation_eigenvectors_results[:,[1,2]].T)
plt.xlabel('Number of eigenvectors (k)')
plt.ylabel('Mean alignment')
plt.ylim([0,1])


fig = plt.figure()
ablation_density_results = [r.mean(0) for r in ablation_density_results]
ablation_density_results = np.array(ablation_density_results)
plt.errorbar(density, ablation_density_results[:,0], yerr = ablation_density_results[:,[1,2]].T)
plt.xlabel('Average distance between sample points (% manifold diam.)')
plt.ylabel('Mean alignment')
plt.ylim([0,1])

