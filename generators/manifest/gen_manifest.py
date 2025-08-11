#!/usr/bin/env python3
import json, sys, re, pathlib
# Super-minimal manifest from tg.h (functions + version)
hdr = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
funcs = re.findall(r'\bint\s+(tg[A-Za-z0-9_]+)\s*\(', hdr)
manifest = {
  "version":"0.0.1",
  "calling_convention":"cdecl",
  "functions": [{"name": f, "ret":"int"} for f in sorted(set(funcs))],
  "enums": {"tg_dtype":["TG_F32"]},
  "notes":"expand parser later; this is a seed for generators"
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(manifest, indent=2))
print(f"Wrote {sys.argv[2]} with {len(funcs)} functions.")
