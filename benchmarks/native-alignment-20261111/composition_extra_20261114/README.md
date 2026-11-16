# Composition supplementary metrics

Frozen predictions; 14 test conditions; 10 V7 types. No training or inference. Each seed scored separately, then equal condition averaging and seed mean/sample SD. Every source reproduces existing composition MAE within 1e-6 pp.

Hellinger = sqrt(0.5 sum((sqrt(pred)-sqrt(truth))^2)), range 0-1, lower better. Standard established distance, not a newly invented metric.

Dominant-type accuracy: whether the predicted largest-proportion cell type matches the true largest-proportion type. Exact ties (absolute tolerance 1e-12) use uniform predicted tie selection and accept any true tied maximum. Report percentage; 14 conditions means 7.14 percentage-point granularity per seed. This does not measure minority-type accuracy; near ties can be unstable. true_top_gap is retained.

Both are exploratory post-hoc additions; no checkpoint selection or claims of statistical significance. Hellinger overlaps conceptually with JSD. PerturbNet and PerturbGen remain legacy versions. Retrieval-only excluded because its separate frozen proportion array has not been verified for this supplement. Single-run baseline SD undefined.

method,n,hellinger_mean,hellinger_sd,dominant_mean,dominant_sd
state,3,0.19382870244597913,0.004281934708676662,54.76190476190476,10.910894511799622
hypo,3,0.19609487945177217,0.017710297063130466,71.42857142857143,0.0
ridge_identity,1,0.23815713485199305,,64.28571428571429,
ridge_embedding,1,0.23815738433660272,,64.28571428571429,
perturbnet,3,0.2748787652710733,0.012192892980780839,28.571428571428573,0.0
cellflow,3,0.27680765415217223,0.0408081544328473,54.76190476190476,8.247860988423227
stack_traincontext,3,0.2929318302941131,0.0063746954273088356,40.476190476190474,14.86904285332952
perturbmean,1,0.3716177567764736,,21.428571428571427,
control,1,0.3787995900131049,,21.428571428571427,
chemcpa,3,0.4413505502770088,0.009319252644067522,21.428571428571427,0.0
cpa,3,0.4732138177560053,0.0674627683985148,26.19047619047619,4.1239304942116135
perturbgen,3,0.4759957301157575,0.05103849526128468,28.571428571428573,12.371791482634839
g2cp,3,0.5727109917427669,0.009097979663712608,14.285714285714285,7.1428571428571415
