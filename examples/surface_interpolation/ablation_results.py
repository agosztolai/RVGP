#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import numpy as np


def process_results(ablation_results, trials):
    results = []
    for i in range(len(ablation_results)):
        gt, pred = ablation_results[i]
        
        alignment = (ablation_results[i][0]*ablation_results[i][1]).sum(1)
        
        alignment_mean = alignment.mean()
        alignment_std = alignment.std()/2
        
        trial = i%trials
        if trial == 0:
            results_condition = np.zeros([trials, 2])
            
        results_condition[trial] = np.array([alignment_mean, alignment_std])
            
        if trial == trials - 1:
            results.append(results_condition)
            
    return results


trials = 10

ablation_eigenvectors = pickle.load(open('ablation_eigenvector_results.pkl', 'rb'))
ablation_eigenvectors_results = process_results(ablation_eigenvectors, trials) 
k = [1,2,5,10,100]

trials = 20

ablation_density = pickle.load(open('ablation_density_results.pkl', 'rb'))
ablation_density_results = process_results(ablation_density, trials) 
density = np.linspace(0.01,0.08,10)

trials = 20

noise_perturbation = pickle.load(open('noise_perturbation_2.pkl', 'rb'))
noise_perturbation_results = process_results(noise_perturbation, trials) 
alpha = np.linspace(0.005,0.05,10)

fig = plt.figure()
ablation_eigenvectors_results = [ablation_eigenvectors_results[i].mean(0) for i in range(len(k))]
ablation_eigenvectors_results = np.array(ablation_eigenvectors_results)
plt.errorbar(k, ablation_eigenvectors_results[:,0], yerr = ablation_eigenvectors_results[:,1].T)
plt.xlabel('Number of eigenvectors (k)')
plt.ylabel('Mean alignment')
plt.ylim([0,1.1])
plt.xscale('log')
plt.savefig('ablation_eigenvectors.svg')


# fig = plt.figure()
# ablation_density_results = [r.mean(0) for r in ablation_density_results]
# ablation_density_results = np.array(ablation_density_results)
# plt.errorbar(100*density, ablation_density_results[:,0], yerr = ablation_density_results[:,1].T)
# plt.xlabel('Average distance between sample points (% manifold diam.)')
# plt.ylabel('Mean alignment')
# plt.ylim([0,1.1])
# plt.savefig('ablation_density.svg')


fig = plt.figure()
noise_perturbation_results = [r.mean(0) for r in noise_perturbation_results]
noise_perturbation_results = np.array(noise_perturbation_results)
plt.errorbar(100*alpha, noise_perturbation_results[:,0], yerr = noise_perturbation_results[:,1].T)
plt.xlabel('Noise amplitude (% manifold diam.)')
plt.ylabel('Mean alignment')
plt.ylim([0,1.1])
plt.savefig('noise_perturbation.svg')


