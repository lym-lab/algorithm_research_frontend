import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
pairs = [
    ('        ours = m.get(f"{key}_avg")', '        value = m.get(f"{key}_avg")'),
    ('        if ours is None:', '        if value is None:'),
    ('            verdict[key] = "ok" if ours >= tol["eer_min"] else f"below {tol[\'eer_min\']}"',
     '            verdict[key] = "ok" if value >= tol["eer_min"] else f"below {tol[\'eer_min\']}"'),
    ('            verdict[key] = "ok" if abs(ours - paper) <= tol[key] else f"off by {ours - paper:+.2f}"',
     '            verdict[key] = "ok" if abs(value - paper) <= tol[key] else f"off by {value - paper:+.2f}"'),
    ('            verdict[key] = f"{ours - paper:+.2f}"', '            verdict[key] = f"{value - paper:+.2f}"'),
    ('            m["verdict"] = judge(args.row, m)',
     '            measured = {r: collect(args.vpc_root, v[0], r) for r, v in targets.TABLE1.items()}\n'
     '            m["verdict"] = judge(args.row, m, measured)'),
]
for old, new in pairs:
    if old not in s:
        print('MISS:', old[:60]); sys.exit(1)
    s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
print('patched', p)
