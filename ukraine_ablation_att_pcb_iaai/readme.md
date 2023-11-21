1) remove everything from bayes_log.json
2) change bayes_sample.sbatch
3) change abm.sbatch
4) change abm.py if needed (flags for ablation)
5) change bayes.py
	a) >=
	b) size of subprocess output condition
	c) new identifier
6) run bayes.sh
7) once run starts change bayes.py b)

