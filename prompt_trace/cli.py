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
      <div class=\"flex flex-wrap items-center gap-3\">
        <div class=\"text-sm text-slate-400\">Rows: <span id=\"rowCount\">0</span></div>
        <div class="inline-flex overflow-hidden rounded-xl border border-slate-700 bg-slate-900">
          <button id="treeExpandAll" class="px-3 py-2 text-sm font-medium text-slate-300">Expand All</button>
          <button id="treeCollapseAll" class="px-3 py-2 text-sm font-medium text-slate-300">Collapse All</button>
        </div>
        <div class=\"inline-flex overflow-hidden rounded-xl border border-slate-700 bg-slate-900\">
          <button id=\"modeTable\" class=\"bg-cyan-600 px-3 py-2 text-sm font-medium text-white\">Table View</button>
          <button id=\"modeTree\" class=\"px-3 py-2 text-sm font-medium text-slate-300\">Compact Tree View</button>
        </div>
      </div>
    </div>

    <div class=\"overflow-hidden rounded-2xl border border-slate-800 bg-slate-900\">
      <div class=\"overflow-x-auto\">
        <table class=\"min-w-full text-left text-sm\">
          <thead class=\"bg-slate-800/80 text-slate-200\">
            <tr>
              <th class=\"px-4 py-3\">ID</th>
              <th class=\"px-4 py-3\">Timestamp</th>
              <th class=\"px-4 py-3\">Run</th>
              <th class=\"px-4 py-3\">Trace</th>
              <th class=\"px-4 py-3\">Parent Trace</th>
              <th class=\"px-4 py-3\">Agent</th>
              <th class=\"px-4 py-3\">Step</th>
              <th class=\"px-4 py-3\">Version</th>
              <th class=\"px-4 py-3\">Model</th>
              <th class=\"px-4 py-3\">Latency (ms)</th>
              <th class=\"px-4 py-3\">Prompt Tokens</th>
              <th class=\"px-4 py-3\">Completion Tokens</th>
              <th class=\"px-4 py-3\">Total Tokens</th>
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
    const treeExpandAll = document.getElementById('treeExpandAll');
    const treeCollapseAll = document.getElementById('treeCollapseAll');
    const modeTable = document.getElementById('modeTable');
    const modeTree = document.getElementById('modeTree');
    let currentMode = 'table';
    const collapsedRuns = new Set();
    const collapsedNodes = new Set();

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

    function filterLogs() {{
      const query = search.value.trim().toLowerCase();
      return logs.filter((log) => {{
        if (!query) return true;
        return [
          log.id,
          log.timestamp,
          log.run_id,
          log.trace_id,
          log.parent_trace_id,
          log.agent_name,
          log.step_name,
          log.version_tag,
          log.model,
          log.latency_ms,
          log.prompt_tokens,
          log.completion_tokens,
          log.total_tokens,
          log.prompt,
          log.response
        ].some((field) => String(field ?? '').toLowerCase().includes(query));
      }});
    }}

    function renderTable(filtered) {{
      tbody.innerHTML = '';
      for (const log of filtered) {{
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/50';
        tr.appendChild(textCell(log.id));
        tr.appendChild(textCell(log.timestamp));
        tr.appendChild(textCell(log.run_id));
        tr.appendChild(textCell(log.trace_id));
        tr.appendChild(textCell(log.parent_trace_id));
        tr.appendChild(textCell(log.agent_name));
        tr.appendChild(textCell(log.step_name));
        tr.appendChild(textCell(log.version_tag));
        tr.appendChild(textCell(log.model));
        tr.appendChild(textCell(log.latency_ms));
        tr.appendChild(textCell(log.prompt_tokens));
        tr.appendChild(textCell(log.completion_tokens));
        tr.appendChild(textCell(log.total_tokens));
        tr.appendChild(blockCell(log.prompt));
        tr.appendChild(blockCell(log.response));
        tbody.appendChild(tr);
      }}
    }}

    function runHeaderRow(runId, count) {{
      const tr = document.createElement('tr');
      tr.className = 'bg-slate-950/70';
      const td = document.createElement('td');
      td.colSpan = 15;
      td.className = 'px-4 py-2';

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'text-left text-xs uppercase tracking-[0.2em] text-cyan-300 hover:text-cyan-200';
      const isCollapsed = collapsedRuns.has(runId);
      const marker = isCollapsed ? '[+]' : '[-]';
      button.textContent = marker + ' Run ' + (runId || 'unscoped') + ' · ' + count + ' step' + (count === 1 ? '' : 's');
      button.addEventListener('click', () => toggleRun(runId));

      td.appendChild(button);
      tr.appendChild(td);
      return tr;
    }}

    function toggleRun(runId) {{
      if (collapsedRuns.has(runId)) {{
        collapsedRuns.delete(runId);
      }} else {{
        collapsedRuns.add(runId);
      }}
      render();
    }}

    function toggleNode(nodeKey) {{
      if (collapsedNodes.has(nodeKey)) {{
        collapsedNodes.delete(nodeKey);
      }} else {{
        collapsedNodes.add(nodeKey);
      }}
      render();
    }}

    function expandAllTree() {{
      collapsedRuns.clear();
      collapsedNodes.clear();
      render();
    }}

    function collapseAllTree() {{
      collapsedNodes.clear();
      const filtered = filterLogs();
      const runSet = new Set();
      for (const log of filtered) {{
        runSet.add(log.run_id || 'unscoped');
      }}
      collapsedRuns.clear();
      for (const runId of runSet) {{
        collapsedRuns.add(runId);
      }}
      render();
    }}

    function updateTreeControls() {{
      const isTreeMode = currentMode === 'tree';
      treeExpandAll.disabled = !isTreeMode;
      treeCollapseAll.disabled = !isTreeMode;
      treeExpandAll.classList.toggle('opacity-50', !isTreeMode);
      treeCollapseAll.classList.toggle('opacity-50', !isTreeMode);
    }}

    function renderTree(filtered) {{
      tbody.innerHTML = '';
      const byRun = new Map();
      for (const log of filtered) {{
        const runId = log.run_id || 'unscoped';
        if (!byRun.has(runId)) byRun.set(runId, []);
        byRun.get(runId).push(log);
      }}

      for (const [runId, runLogs] of byRun.entries()) {{
        const traceMap = new Map();
        const childrenMap = new Map();
        for (const log of runLogs) {{
          traceMap.set(log.trace_id, log);
          const parentKey = log.parent_trace_id || '__root__';
          if (!childrenMap.has(parentKey)) childrenMap.set(parentKey, []);
          childrenMap.get(parentKey).push(log);
        }}

        const roots = runLogs.filter((log) => !log.parent_trace_id || !traceMap.has(log.parent_trace_id));
        tbody.appendChild(runHeaderRow(runId, runLogs.length));

        if (collapsedRuns.has(runId)) {{
          continue;
        }}

        const visit = (log, depth) => {{
          const nodeKey = String(log.trace_id || log.id || 'node');
          const nodeChildren = childrenMap.get(log.trace_id) || [];
          const hasChildren = nodeChildren.length > 0;
          const nodeCollapsed = collapsedNodes.has(nodeKey);

          const tr = document.createElement('tr');
          tr.className = 'hover:bg-slate-800/50';
          tr.appendChild(textCell(log.id));
          tr.appendChild(textCell(log.timestamp));
          tr.appendChild(textCell(log.run_id));
          tr.appendChild(textCell(log.trace_id));
          tr.appendChild(textCell(log.parent_trace_id));
          tr.appendChild(textCell(log.agent_name));

          const stepTd = document.createElement('td');
          stepTd.className = 'px-4 py-3 align-top text-slate-300';

          const branchWrap = document.createElement('span');
          branchWrap.className = 'inline-flex items-center';

          const indentSpan = document.createElement('span');
          indentSpan.className = 'font-mono text-xs text-slate-500';
          indentSpan.textContent = depth > 0 ? '  '.repeat(depth) : '';
          branchWrap.appendChild(indentSpan);

          if (hasChildren) {{
            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'mr-1 rounded px-1 font-mono text-xs text-cyan-300 hover:bg-slate-800';
            toggle.textContent = nodeCollapsed ? '[+]' : '[-]';
            toggle.addEventListener('click', () => toggleNode(nodeKey));
            branchWrap.appendChild(toggle);
          }} else {{
            const bullet = document.createElement('span');
            bullet.className = 'mr-1 font-mono text-xs text-slate-500';
            bullet.textContent = depth > 0 ? '└─' : '•';
            branchWrap.appendChild(bullet);
          }}

          const nameSpan = document.createElement('span');
          nameSpan.textContent = String(log.step_name ?? '');
          branchWrap.appendChild(nameSpan);
          stepTd.appendChild(branchWrap);
          tr.appendChild(stepTd);

          tr.appendChild(textCell(log.version_tag));
          tr.appendChild(textCell(log.model));
          tr.appendChild(textCell(log.latency_ms));
          tr.appendChild(textCell(log.prompt_tokens));
          tr.appendChild(textCell(log.completion_tokens));
          tr.appendChild(textCell(log.total_tokens));
          tr.appendChild(blockCell(log.prompt));
          tr.appendChild(blockCell(log.response));
          tbody.appendChild(tr);

          if (nodeCollapsed) {{
            return;
          }}

          for (const child of nodeChildren) {{
            visit(child, depth + 1);
          }}
        }};

        for (const root of roots) {{
          visit(root, 0);
        }}
      }}
    }}

    function render() {{
      const filtered = filterLogs();
      rowCount.textContent = String(filtered.length);
      if (currentMode === 'tree') {{
        renderTree(filtered);
      }} else {{
        renderTable(filtered);
      }}
    }}

    modeTable.addEventListener('click', () => {{
      currentMode = 'table';
      modeTable.classList.add('bg-cyan-600', 'text-white');
      modeTable.classList.remove('text-slate-300');
      modeTree.classList.remove('bg-cyan-600', 'text-white');
      modeTree.classList.add('text-slate-300');
      updateTreeControls();
      render();
    }});

    modeTree.addEventListener('click', () => {{
      currentMode = 'tree';
      modeTree.classList.add('bg-cyan-600', 'text-white');
      modeTree.classList.remove('text-slate-300');
      modeTable.classList.remove('bg-cyan-600', 'text-white');
      modeTable.classList.add('text-slate-300');
      updateTreeControls();
      render();
    }});

    treeExpandAll.addEventListener('click', expandAllTree);
    treeCollapseAll.addEventListener('click', collapseAllTree);

    search.addEventListener('input', render);
    updateTreeControls();
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
