# Read-only status: runner, failures, footprint coverage, GPUs.
date; python3 jobs/runner.py status 2>&1 | grep -v "^done"
for f in $(ls -t jobs/logs/*.log | head -40); do grep -l "Traceback\|Error" $f; done 2>/dev/null | head
tail -3 data/neurons/1.4b_s0_step143000/footprint/footprint.log
python3 -c "
import json; m=json.load(open('data/neurons/1.4b_s0_step143000/footprint/footprint.json'))
import collections; d=collections.defaultdict(list)
[d[v['class']].append(v['n']) for v in m['tokens'].values()]
print('streamed', m['tokens_streamed']); [print(k, len(v), 'min', min(v), 'median', sorted(v)[len(v)//2]) for k,v in d.items()]"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; df -h data | tail -1
