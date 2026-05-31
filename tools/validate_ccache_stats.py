import os
import pathlib
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

# Print only Cytnx-related ccache log entries, then remove the log to avoid carry-over.
log_path = os.getenv('CCACHE_LOGFILE', '.ccache/cytnx-ccache.log')
log_file = pathlib.Path(log_path)
repo_root = pathlib.Path.cwd().resolve()
try:
    if log_file.exists():
        print(f'filtered_ccache_log={log_file}')
        lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()

        # Detect the section delimiter by selecting the most repeated line in the first chunk.
        head = lines[:400]
        counts = {}
        for ln in head:
            key = ln.strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        sep = max(counts, key=counts.get) if counts else ''

        cytnx_hits = []
        for i, ln in enumerate(lines):
            low = ln.lower()
            if '/cytnx/' in low or str(repo_root).lower() in low:
                cytnx_hits.append(i)

        if not cytnx_hits:
            print('ccache_log_no_cytnx_entries_found')
        elif not sep:
            for i in cytnx_hits:
                print(lines[i])
        else:
            sections = set()
            for i in cytnx_hits:
                start = i
                while start > 0 and lines[start].strip() != sep:
                    start -= 1
                end = i
                while end + 1 < len(lines) and lines[end + 1].strip() != sep:
                    end += 1
                sections.add((start, end))

            for start, end in sorted(sections):
                for ln in lines[start:end + 1]:
                    print(ln)
                if end + 1 < len(lines):
                    print(lines[end + 1])
    else:
        print(f'ccache_log_missing={log_file}')
finally:
    if log_file.exists():
        log_file.unlink()

# Reset counters so the next Python-version build observes fresh, per-build stats.
subprocess.check_call(['ccache', '--zero-stats'])

if total <= 0:
    raise SystemExit('No ccache hits detected for this python-version build.')
