from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .core import get_logs


def _build_html(logs: List[Dict[str, Any]], db_path: str) -> str:
    logs_json = json.dumps(logs, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang=\"en\" class=\"dark\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>PromptTrace Dashboard</title>
  <script src=\"https://cdn.tailwindcss.com\"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            slate: tailwind.colors.slate,
            cyan: tailwind.colors.cyan,
            emerald: tailwind.colors.emerald
          }}
        }}
      }}
    }};
  </script>
</head>
<body class=\"min-h-screen bg-slate-950 text-slate-100\">
  <div class=\"mx-auto max-w-7xl p-6\">
    <div class=\"mb-8 rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl shadow-cyan-950/20\">
      <h1 class=\"text-3xl font-bold tracking-tight text-cyan-300\">PromptTrace</h1>
      <p class=\"mt-2 text-slate-300\">Local trace dashboard for prompt debugging and version-aware iteration.</p>
      <p class=\"mt-1 text-xs text-slate-400\">Source DB: {db_path}</p>
    </div>

    <div class=\"mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between\">
      <input id=\"search\" type=\"text\" placeholder=\"Search prompt, response, model, or version...\"
        class=\"w-full md:max-w-xl rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500\" />
      <div class=\"text-sm text-slate-400\">Rows: <span id=\"rowCount\">0</span></div>
    </div>

    <div class=\"overflow-hidden rounded-2xl border border-slate-800 bg-slate-900\">
      <div class=\"overflow-x-auto\">
        <table class=\"min-w-full text-left text-sm\">
          <thead class=\"bg-slate-800/80 text-slate-200\">
            <tr>
              <th class=\"px-4 py-3\">ID</th>
              <th class=\"px-4 py-3\">Timestamp</th>
              <th class=\"px-4 py-3\">Version</th>
              <th class=\"px-4 py-3\">Model</th>
              <th class=\"px-4 py-3\">Latency (ms)</th>
              <th class=\"px-4 py-3\">Prompt</th>
              <th class=\"px-4 py-3\">Response</th>
            </tr>
          </thead>
          <tbody id=\"rows\" class=\"divide-y divide-slate-800\"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const logs = {logs_json};
    const tbody = document.getElementById('rows');
    const rowCount = document.getElementById('rowCount');
    const search = document.getElementById('search');

    function textCell(value) {{
      const td = document.createElement('td');
      td.className = 'px-4 py-3 align-top text-slate-300';
      td.textContent = value == null ? '' : String(value);
      return td;
    }}

    function blockCell(value) {{
      const td = document.createElement('td');
      td.className = 'px-4 py-3 align-top';
      const pre = document.createElement('pre');
      pre.className = 'max-w-xl overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950/70 p-2 text-xs text-emerald-300';
      pre.textContent = value == null ? '' : String(value);
      td.appendChild(pre);
      return td;
    }}

    function render() {{
      const query = search.value.trim().toLowerCase();
      tbody.innerHTML = '';

      const filtered = logs.filter((log) => {{
        if (!query) return true;
        return [
          log.id,
          log.timestamp,
          log.version_tag,
          log.model,
          log.latency_ms,
          log.prompt,
          log.response
        ].some((field) => String(field ?? '').toLowerCase().includes(query));
      }});

      for (const log of filtered) {{
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/50';
        tr.appendChild(textCell(log.id));
        tr.appendChild(textCell(log.timestamp));
        tr.appendChild(textCell(log.version_tag));
        tr.appendChild(textCell(log.model));
        tr.appendChild(textCell(log.latency_ms));
        tr.appendChild(blockCell(log.prompt));
        tr.appendChild(blockCell(log.response));
        tbody.appendChild(tr);
      }}

      rowCount.textContent = String(filtered.length);
    }}

    search.addEventListener('input', render);
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone PromptTrace HTML dashboard")
    parser.add_argument("--db-path", default="./prompt_trace.db", help="Path to SQLite database")
    parser.add_argument("--output", default="index.html", help="Output HTML file path")
    args = parser.parse_args()

    logs = get_logs(args.db_path)
    html = _build_html(logs, args.db_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Generated dashboard with {len(logs)} logs at {output_path.resolve()}")


if __name__ == "__main__":
    main()
