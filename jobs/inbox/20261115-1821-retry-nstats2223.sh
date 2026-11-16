# Retry nstats-L22-23 after the model-offload fix.
git log -1 --format="%h %s"
python3 jobs/runner.py retry nstats-L22-23
