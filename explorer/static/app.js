"use strict";

const state = { runs: [], metrics: [], tuning: [], groups: [] };
const byId = id => document.getElementById(id);

function clear(node) { node.replaceChildren(); }
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function option(value, text = value) { const node = element("option", text); node.value = value; return node; }
async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

document.querySelectorAll(".nav").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".nav,.view").forEach(node => node.classList.remove("active"));
  button.classList.add("active");
  byId(button.dataset.view).classList.add("active");
}));

function populateFilters() {
  const group = byId("group-filter"); clear(group); group.append(option("", "No experiment group"));
  state.groups.filter(item => item.status === "available").forEach(item => group.append(option(item.name, `${item.name} · ${item.split_role}`)));
  const benchmark = byId("benchmark-filter");
  clear(benchmark);
  [...new Set(state.metrics.map(row => row.benchmark))].sort().forEach(value => benchmark.append(option(value)));
  const method = byId("method-filter"); clear(method); method.append(option("", "All methods"));
  [...new Set(state.metrics.map(row => row.method))].sort().forEach(value => method.append(option(value)));
  updateMetricFilters();
  for (const id of ["run-a", "run-b"]) {
    const select = byId(id); clear(select);
    state.runs.forEach(run => select.append(option(run.key, `${run.method} · ${run.benchmark} · ${run.run_id}`)));
  }
  const baseline = state.runs.findIndex(run => run.method === "baseline");
  if (baseline >= 0) byId("run-a").selectedIndex = baseline;
  const steered = state.runs.findIndex(run => run.method !== "baseline");
  if (steered >= 0) byId("run-b").selectedIndex = steered;
}

function updateMetricFilters() {
  const selectedBenchmark = byId("benchmark-filter").value;
  const matching = state.metrics.filter(row => row.benchmark === selectedBenchmark);
  for (const [id, values] of [["metric-filter", matching.map(row => row.metric)], ["subgroup-filter", matching.map(row => row.subgroup)]]) {
    const select = byId(id); const previous = select.value; clear(select);
    [...new Set(values)].sort().forEach(value => select.append(option(value)));
    if ([...select.options].some(item => item.value === previous)) select.value = previous;
  }
  renderOverview();
}

function renderOverview() {
  const selectedGroup = state.groups.find(group => group.name === byId("group-filter").value);
  const allowedRuns = selectedGroup ? new Set(selectedGroup.points.flatMap(point => [point.bias.run_key, point.capability.run_key])) : null;
  const selectedMethod = byId("method-filter").value;
  const axisRows = state.metrics.filter(row => (!allowedRuns || allowedRuns.has(row.run_key)) && row.benchmark === byId("benchmark-filter").value && row.metric === byId("metric-filter").value && row.subgroup === byId("subgroup-filter").value);
  const selected = axisRows.filter(row => !selectedMethod || row.method === selectedMethod);
  const baseline = selectedGroup ? axisRows.find(row => row.method === "baseline" && row.value !== null) : null;
  const body = byId("metrics-body"); clear(body);
  selected.forEach(row => {
    const tr = document.createElement("tr");
    const delta = baseline && row.value !== null ? row.value - baseline.value : null;
    [row.method, row.strength ?? "—", row.metric, row.value ?? "undefined", delta === null ? "—" : format(delta), row.n, row.run_key].forEach(value => tr.append(element("td", value)));
    body.append(tr);
  });
  const unavailable = state.groups.filter(group => group.status !== "available").map(group => `${group.name}: ${group.reason}`);
  byId("overview-message").textContent = selectedGroup ? `${selectedGroup.name} · ${selectedGroup.split_role} · ${selectedGroup.model_name}@${selectedGroup.model_revision.slice(0, 8)}` : `Metrics are listed individually. Select a validated experiment group to compare methods.${unavailable.length ? ` Unavailable groups — ${unavailable.join(" ")}` : ""}`;
  renderTradeoff(byId("tradeoff-chart"), selectedGroup);
  renderBars(byId("metric-chart"), selectedGroup ? selected.filter(row => row.value !== null) : []);
}

function renderBars(svg, rows) {
  clear(svg); svg.setAttribute("viewBox", "0 0 800 220");
  if (!rows.length) { const text = document.createElementNS("http://www.w3.org/2000/svg", "text"); text.setAttribute("class", "label"); text.setAttribute("x", "20"); text.setAttribute("y", "35"); text.textContent = "No numeric metric values available"; svg.append(text); return; }
  const values = rows.map(row => row.value); const low = Math.min(0, ...values); const high = Math.max(0, ...values); const span = high - low || 1;
  const zero = 150 + 620 * (0 - low) / span;
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line"); axis.setAttribute("x1", String(zero)); axis.setAttribute("x2", String(zero)); axis.setAttribute("y1", "5"); axis.setAttribute("y2", "210"); axis.setAttribute("class", "axis"); svg.append(axis);
  rows.forEach((row, index) => {
    const endpoint = 150 + 620 * (row.value - low) / span; const width = Math.abs(endpoint - zero); const y = 18 + index * Math.min(42, 180 / rows.length);
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(Math.min(zero, endpoint))); rect.setAttribute("y", String(y)); rect.setAttribute("width", String(width)); rect.setAttribute("height", "22"); rect.setAttribute("fill", row.method === "baseline" ? "#8b98a5" : "#55d6be"); svg.append(rect);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text"); label.setAttribute("x", "8"); label.setAttribute("y", String(y + 16)); label.setAttribute("class", "label"); label.textContent = `${row.method} ${row.strength ?? ""}`; svg.append(label);
  });
}

function renderTradeoff(svg, group) {
  clear(svg); svg.setAttribute("viewBox", "0 0 800 300");
  if (!group) { const text = document.createElementNS("http://www.w3.org/2000/svg", "text"); text.setAttribute("class", "label"); text.setAttribute("x", "20"); text.setAttribute("y", "35"); text.textContent = "Select a configured experiment group"; svg.append(text); return; }
  const xs = group.points.map(point => point.bias.value); const ys = group.points.map(point => point.capability.value); const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y0 = Math.min(...ys); const y1 = Math.max(...ys);
  const px = point => 55 + 690 * (point.bias.value - x0) / (x1 - x0 || 1); const py = point => 255 - 220 * (point.capability.value - y0) / (y1 - y0 || 1); const colors = { caa: "#f4b860", icv: "#55d6be", sae: "#9b8afb", reft_r1: "#ff7b72" }; const baseline = group.points.find(point => point.method === "baseline");
  [...new Set(group.points.filter(point => point.method !== "baseline").map(point => point.method))].forEach(method => { const ordered = [baseline, ...group.points.filter(point => point.method === method)].filter(Boolean).sort((a, b) => (a.strength ?? 0) - (b.strength ?? 0)); const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); path.setAttribute("points", ordered.map(point => `${px(point)},${py(point)}`).join(" ")); path.setAttribute("fill", "none"); path.setAttribute("stroke", colors[method] || "#55d6be"); path.setAttribute("stroke-width", "2"); svg.append(path); });
  group.points.forEach(point => { const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(px(point))); dot.setAttribute("cy", String(py(point))); dot.setAttribute("r", "7"); dot.setAttribute("fill", point.method === "baseline" ? "#8b98a5" : colors[point.method] || "#55d6be"); dot.setAttribute("class", "dot"); const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = `${point.method} ${point.strength ?? ""} · bias ${point.bias.value} · capability ${point.capability.value}`; dot.append(title); svg.append(dot); });
  const xlabel = document.createElementNS("http://www.w3.org/2000/svg", "text"); xlabel.setAttribute("x", "55"); xlabel.setAttribute("y", "290"); xlabel.setAttribute("class", "label"); xlabel.textContent = `${group.points[0].bias.benchmark}: ${group.points[0].bias.metric} (lower preferred)`; svg.append(xlabel);
  const ylabel = document.createElementNS("http://www.w3.org/2000/svg", "text"); ylabel.setAttribute("x", "55"); ylabel.setAttribute("y", "20"); ylabel.setAttribute("class", "label"); ylabel.textContent = `${group.points[0].capability.benchmark}: ${group.points[0].capability.metric} (higher preferred)`; svg.append(ylabel);
}

function renderStrength() {
  const target = byId("strength-content"); clear(target);
  if (!state.tuning.length) target.append(element("div", "No saved tuning summaries are available. Validated evaluation curves remain available below.", "unavailable"));
  state.tuning.forEach(item => {
    const card = element("article", undefined, "card"); card.append(element("h3", item.run_id));
    card.append(element("p", `Status: ${item.summary.status || "unknown"}`, "meta"));
    const selections = item.summary.selections && typeof item.summary.selections === "object" ? item.summary.selections : {};
    Object.entries(selections).forEach(([method, strength]) => card.append(element("p", `Development-selected ${method} strength: ${strength}`, "eyebrow")));
    const aggregates = Array.isArray(item.summary.aggregates) ? item.summary.aggregates : [];
    aggregates.forEach(row => card.append(element("p", `${row.method} · strength ${row.strength}: balanced ${format(row.mean_balanced_score)}, bias avoidance ${format(row.mean_bias_avoidance)}, quality ${format(row.mean_response_quality)} (n=${row.n})`)));
    target.append(card);
  });
  state.groups.forEach(group => {
    const card = element("article", undefined, "card");
    card.append(element("h3", `${group.name} · ${group.split_role || "unknown split"}`));
    if (group.status !== "available") { card.append(element("div", group.reason, "unavailable")); target.append(card); return; }
    const baseline = group.points.find(point => point.method === "baseline");
    const table = document.createElement("table"); const body = document.createElement("tbody");
    group.points.slice().sort((a, b) => (a.strength ?? 0) - (b.strength ?? 0)).forEach(point => {
      const tr = document.createElement("tr");
      const biasDelta = baseline ? point.bias.value - baseline.bias.value : null; const capabilityDelta = baseline ? point.capability.value - baseline.capability.value : null;
      [point.method, `strength ${point.strength ?? 0}`, `bias ${format(point.bias.value)} (${biasDelta === null ? "—" : format(biasDelta)})`, `capability ${format(point.capability.value)} (${capabilityDelta === null ? "—" : format(capabilityDelta)})`].forEach(value => tr.append(element("td", value))); body.append(tr);
    });
    table.append(body); card.append(table);
    const repeats = new Map(); group.points.forEach(point => { const key = `${point.method}/${point.strength ?? 0}`; const current = repeats.get(key) || []; current.push(point); repeats.set(key, current); });
    [...repeats.entries()].forEach(([key, rows]) => { const seeds = new Set(rows.map(row => row.seed)); if (seeds.size < 2) return; const bias = rows.map(row => row.bias.value); const capability = rows.map(row => row.capability.value); card.append(element("p", `${key} · ${seeds.size} seeds · bias range ${format(Math.min(...bias))}–${format(Math.max(...bias))} · capability range ${format(Math.min(...capability))}–${format(Math.max(...capability))}`, "meta")); });
    const biasChart = document.createElementNS("http://www.w3.org/2000/svg", "svg"); renderCurve(biasChart, group.points, "bias"); card.append(biasChart);
    const capabilityChart = document.createElementNS("http://www.w3.org/2000/svg", "svg"); renderCurve(capabilityChart, group.points, "capability"); card.append(capabilityChart);
    const trajectory = document.createElementNS("http://www.w3.org/2000/svg", "svg"); renderTradeoff(trajectory, group); card.append(trajectory);
    target.append(card);
  });
}
function format(value) { return typeof value === "number" ? value.toFixed(3) : "unavailable"; }

function renderCurve(svg, points, axis) {
  svg.setAttribute("viewBox", "0 0 700 220"); const rows = points.filter(point => typeof point[axis].value === "number");
  if (!rows.length) return;
  const xs = rows.map(point => point.strength ?? 0); const ys = rows.map(point => point[axis].value); const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y0 = Math.min(...ys); const y1 = Math.max(...ys);
  const colors = { caa: "#f4b860", icv: "#55d6be", sae: "#9b8afb", reft_r1: "#ff7b72" }; const baseline = rows.find(point => point.method === "baseline");
  [...new Set(rows.filter(point => point.method !== "baseline").map(point => point.method))].forEach(method => {
    const ordered = [baseline, ...rows.filter(point => point.method === method)].filter(Boolean).sort((a, b) => (a.strength ?? 0) - (b.strength ?? 0));
    const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); path.setAttribute("fill", "none"); path.setAttribute("stroke", colors[method] || "#55d6be"); path.setAttribute("stroke-width", "2"); path.setAttribute("points", ordered.map(point => `${55 + 600 * ((point.strength ?? 0) - x0) / (x1 - x0 || 1)},${180 - 140 * (point[axis].value - y0) / (y1 - y0 || 1)}`).join(" ")); svg.append(path);
    ordered.forEach(point => { const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(55 + 600 * ((point.strength ?? 0) - x0) / (x1 - x0 || 1))); dot.setAttribute("cy", String(180 - 140 * (point[axis].value - y0) / (y1 - y0 || 1))); dot.setAttribute("r", "5"); dot.setAttribute("fill", point.method === "baseline" ? "#8b98a5" : colors[method] || "#55d6be"); svg.append(dot); });
  });
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text"); label.setAttribute("x", "15"); label.setAttribute("y", "20"); label.setAttribute("class", "label"); label.textContent = `${axis} against strength`; svg.append(label);
}

async function compareRuns() {
  const message = byId("generation-message"); const target = byId("generation-content"); clear(target); message.textContent = "Loading comparison…";
  try {
    const result = await api("/api/generations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_keys: [byId("run-a").value, byId("run-b").value] }) });
    message.textContent = `${result.benchmark} · ${result.split}. ${result.warnings.length ? result.warnings.join(" ") : result.prompt_status === "available" ? "Runs pass generation and exact prompt-coverage checks." : result.prompt_reason}`;
    const filters = element("div", undefined, "filters"); const category = document.createElement("select"); category.append(option("", "All categories")); [...new Set(result.examples.map(example => example.metadata?.category).filter(Boolean))].sort().forEach(value => category.append(option(value))); const categoryLabel = element("label", "Category or subgroup"); categoryLabel.append(category); filters.append(categoryLabel);
    const exampleSelect = document.createElement("select"); exampleSelect.append(option("", "All examples")); result.examples.forEach(example => exampleSelect.append(option(example.example_id))); const exampleLabel = element("label", "Example"); exampleLabel.append(exampleSelect); filters.append(exampleLabel); target.append(filters);
    const lanes = Object.fromEntries(result.lanes.map(lane => [lane.method, lane]));
    const list = element("div"); target.append(list);
    const render = () => { clear(list); result.examples.filter(example => (!category.value || example.metadata?.category === category.value) && (!exampleSelect.value || example.example_id === exampleSelect.value)).forEach(example => {
      const article = element("article", undefined, "generation"); const heading = element("h3", example.example_id); article.append(heading);
      if (example.prompt) article.append(element("div", example.prompt, "output"));
      if (example.metadata && Object.keys(example.metadata).length) article.append(element("p", JSON.stringify(example.metadata), "meta"));
      Object.entries(example.outputs).forEach(([method, output]) => { const lane = element("div"); const identity = lanes[method]; lane.append(element("strong", `${method} · strength ${identity?.strength ?? 0}`)); lane.append(element("div", output, "output")); article.append(lane); });
      list.append(article);
    }); };
    category.addEventListener("change", render); exampleSelect.addEventListener("change", render); render();
  } catch (error) { message.textContent = error.message; message.classList.add("error"); }
}

async function renderControls() {
  const target = byId("controls-content"); clear(target);
  try {
    const result = await api("/api/controls");
    if (result.status !== "available") { target.append(element("div", result.reason, "unavailable")); return; }
    result.records.forEach(record => {
      const card = element("article", undefined, "card"); card.append(element("h3", `${record.observed.method} · ${record.metric.name}`));
      if (record.summary.status !== "available") card.append(element("div", `Insufficient controls: ${record.summary.count}`, "unavailable"));
      else card.append(element("p", `${record.family}. ${record.summary.count} controls. Empirical percentile: ${format(record.summary.percentile)}.`));
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); renderControlDistribution(svg, record); card.append(svg);
      card.append(element("p", `Observed ${format(record.observed.effect)} · run ${record.observed.run_id}`, "meta"));
      card.append(element("p", `Matching: ${JSON.stringify(record.matching)}`, "meta"));
      card.append(element("p", `Controls: ${record.controls.map(item => item.run_id).join(", ")}`, "meta")); target.append(card);
    });
  } catch (error) { target.append(element("div", error.message, "error")); }
}

function renderControlDistribution(svg, record) {
  svg.setAttribute("viewBox", "0 0 700 150"); const values = record.controls.map(item => item.effect); const all = [...values, record.observed.effect]; const low = Math.min(...all); const high = Math.max(...all); const x = value => 40 + 620 * (value - low) / (high - low || 1);
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line"); axis.setAttribute("x1", "40"); axis.setAttribute("x2", "660"); axis.setAttribute("y1", "90"); axis.setAttribute("y2", "90"); axis.setAttribute("class", "axis"); svg.append(axis);
  values.forEach((value, index) => { const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(x(value))); dot.setAttribute("cy", String(80 - (index % 4) * 13)); dot.setAttribute("r", "5"); dot.setAttribute("fill", "#8b98a5"); svg.append(dot); });
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "line"); marker.setAttribute("x1", String(x(record.observed.effect))); marker.setAttribute("x2", String(x(record.observed.effect))); marker.setAttribute("y1", "20"); marker.setAttribute("y2", "105"); marker.setAttribute("stroke", "#f4b860"); marker.setAttribute("stroke-width", "3"); svg.append(marker);
}

async function renderRepresentation() {
  const target = byId("representation-content"); clear(target);
  try {
    const result = await api("/api/representation");
    if (result.status !== "available") { target.append(element("div", result.reason, "unavailable")); return; }
    result.bundles.forEach(bundle => renderRepresentationBundle(target, bundle));
  } catch (error) { target.append(element("div", error.message, "error")); }
}

function renderRepresentationBundle(target, bundle) {
  const card = element("article", undefined, "card"); card.append(element("h3", `${bundle.config.model.name}@${bundle.config.model.revision} · ${bundle.config.source_split}`));
  const filters = element("div", undefined, "filters"); const definitions = [["method", "Method"], ["strength", "Strength"], ["category", "Category"], ["layer", "Layer"], ["example_id", "Example"]]; const selects = {};
  definitions.forEach(([field, label]) => { const wrapper = element("label", label); const select = document.createElement("select"); select.append(option("", "All")); [...new Set(bundle.records.map(row => String(row[field])))].sort().forEach(value => select.append(option(value))); wrapper.append(select); filters.append(wrapper); selects[field] = select; }); card.append(filters);
  const summary = element("p", undefined, "meta"); card.append(summary); const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); card.append(svg);
  const update = () => { const rows = bundle.records.filter(row => definitions.every(([field]) => !selects[field].value || String(row[field]) === selects[field].value)); renderScatter(svg, rows); const paired = rows.filter(row => typeof row.activation_displacement === "number" && row.state !== "baseline"); summary.textContent = `${paired.length} paired interventions · mean activation displacement ${paired.length ? format(paired.reduce((total, row) => total + row.activation_displacement, 0) / paired.length) : "unavailable"} · reference baseline ${bundle.config.baseline_run_id} · layer ${bundle.records[0].layer} · ${bundle.records[0].hook} · ${bundle.records[0].token_selection}`; };
  Object.values(selects).forEach(select => select.addEventListener("change", update)); update(); card.append(element("p", `Shared PCA explained variance: ${bundle.explained_variance_ratio.map(format).join(", ")}`, "meta")); target.append(card);
}

function renderScatter(svg, rows) {
  clear(svg);
  svg.setAttribute("viewBox", "0 0 700 340");
  if (!rows.length) { const text = document.createElementNS("http://www.w3.org/2000/svg", "text"); text.setAttribute("x", "20"); text.setAttribute("y", "35"); text.setAttribute("class", "label"); text.textContent = "No captures match these filters"; svg.append(text); return; }
  const xs = rows.flatMap(row => typeof row.baseline_x === "number" ? [row.x, row.baseline_x] : [row.x]); const ys = rows.flatMap(row => typeof row.baseline_y === "number" ? [row.y, row.baseline_y] : [row.y]); const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y0 = Math.min(...ys); const y1 = Math.max(...ys);
  const cx = value => 30 + 640 * (value - x0) / (x1 - x0 || 1); const cy = value => 310 - 280 * (value - y0) / (y1 - y0 || 1); const colors = { baseline: "#8b98a5", caa: "#f4b860", icv: "#55d6be", sae: "#9b8afb", reft_r1: "#ff7b72" };
  const definitions = document.createElementNS("http://www.w3.org/2000/svg", "defs"); const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker"); marker.setAttribute("id", "arrowhead"); marker.setAttribute("markerWidth", "7"); marker.setAttribute("markerHeight", "7"); marker.setAttribute("refX", "6"); marker.setAttribute("refY", "3"); marker.setAttribute("orient", "auto"); const arrow = document.createElementNS("http://www.w3.org/2000/svg", "path"); arrow.setAttribute("d", "M0,0 L0,6 L7,3 z"); arrow.setAttribute("fill", "#8b98a5"); marker.append(arrow); definitions.append(marker); svg.append(definitions);
  rows.filter(row => row.state !== "baseline" && typeof row.baseline_x === "number").forEach(row => { const line = document.createElementNS("http://www.w3.org/2000/svg", "line"); line.setAttribute("x1", String(cx(row.baseline_x))); line.setAttribute("y1", String(cy(row.baseline_y))); line.setAttribute("x2", String(cx(row.x))); line.setAttribute("y2", String(cy(row.y))); line.setAttribute("stroke", colors[row.method] || "#55d6be"); line.setAttribute("opacity", ".65"); line.setAttribute("marker-end", "url(#arrowhead)"); svg.append(line); });
  rows.forEach(row => { const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(cx(row.x))); dot.setAttribute("cy", String(cy(row.y))); dot.setAttribute("r", "6"); dot.setAttribute("fill", colors[row.method] || "#55d6be"); dot.setAttribute("class", "dot"); const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = `${row.method} · ${row.example_id} · activation displacement ${format(row.activation_displacement)}`; dot.append(title); svg.append(dot); });
}

async function start() {
  try {
    const [status, runs, metrics, tuning, groups] = await Promise.all([api("/api/status"), api("/api/runs"), api("/api/metrics"), api("/api/tuning"), api("/api/groups")]);
    Object.assign(state, { runs, metrics, tuning, groups });
    byId("status").textContent = `${status.run_count} completed runs · CPU-only viewer`;
    populateFilters(); renderStrength(); await Promise.all([renderControls(), renderRepresentation()]);
  } catch (error) { byId("status").textContent = error.message; byId("status").classList.add("error"); }
}

byId("benchmark-filter").addEventListener("change", updateMetricFilters);
byId("group-filter").addEventListener("change", renderOverview);
byId("metric-filter").addEventListener("change", renderOverview);
byId("subgroup-filter").addEventListener("change", renderOverview);
byId("method-filter").addEventListener("change", renderOverview);
byId("compare-runs").addEventListener("click", compareRuns);
start();

