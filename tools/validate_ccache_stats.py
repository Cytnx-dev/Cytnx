import subprocess

raw = subprocess.check_output(['ccache', '--print-stats'], text=True)
stats = {}
for line in raw.splitlines():
    if '\t' not in line:
        continue
    key, value = line.split('\t', 1)
    stats[key.strip()] = value.strip()

def to_num(v: str) -> float:
    token = v.split()[0] if v else '0'
    try:
        return float(token)
    except Exception:
        return 0.0

direct = to_num(stats.get('direct_cache_hit', stats.get('cache_hit_direct', '0')))
preprocessed = to_num(stats.get('preprocessed_cache_hit', stats.get('cache_hit_preprocessed', '0')))
total = direct + preprocessed

print(f'cache_hit_direct={direct}')
print(f'cache_hit_preprocessed={preprocessed}')
print(f'cache_hit_total={total}')
if total <= 0:
    raise SystemExit('No ccache hits detected for this python-version build.')
