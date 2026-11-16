# Read-only: why nstats-L22-23 failed.
for f in jobs/logs/nstats-L22-23.1.log jobs/logs/nstats-L22-23.2.log; do echo "== $f"; grep -v "Loading weights" $f | tail -25; done
ls -la data/neurons/1.4b_s0_step143000/ | grep -E "L2[0-3]|tokcounts" ; df -h ~ | tail -1
