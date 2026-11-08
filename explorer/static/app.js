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
  const selected = state.metrics.filter(row => (!allowedRuns || allowedRuns.has(row.run_key)) && row.benchmark === byId("benchmark-filter").value && row.metric === byId("metric-filter").value && row.subgroup === byId("subgroup-filter").value);
  const body = byId("metrics-body"); clear(body);
  selected.forEach(row => {
    const tr = document.createElement("tr");
    [row.method, row.strength ?? "—", row.metric, row.value ?? "undefined", row.n, row.run_key].forEach(value => tr.append(element("td", value)));
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
  group.points.forEach(point => { const x = 55 + 690 * (point.bias.value - x0) / (x1 - x0 || 1); const y = 255 - 220 * (point.capability.value - y0) / (y1 - y0 || 1); const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(x)); dot.setAttribute("cy", String(y)); dot.setAttribute("r", "7"); dot.setAttribute("fill", point.method === "baseline" ? "#8b98a5" : "#55d6be"); dot.setAttribute("class", "dot"); const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = `${point.method} ${point.strength ?? ""} · bias ${point.bias.value} · capability ${point.capability.value}`; dot.append(title); svg.append(dot); });
  const xlabel = document.createElementNS("http://www.w3.org/2000/svg", "text"); xlabel.setAttribute("x", "55"); xlabel.setAttribute("y", "290"); xlabel.setAttribute("class", "label"); xlabel.textContent = `${group.points[0].bias.benchmark}: ${group.points[0].bias.metric} (lower preferred)`; svg.append(xlabel);
  const ylabel = document.createElementNS("http://www.w3.org/2000/svg", "text"); ylabel.setAttribute("x", "55"); ylabel.setAttribute("y", "20"); ylabel.setAttribute("class", "label"); ylabel.textContent = `${group.points[0].capability.benchmark}: ${group.points[0].capability.metric} (higher preferred)`; svg.append(ylabel);
}

function renderStrength() {
  const target = byId("strength-content"); clear(target);
  if (!state.tuning.length) { target.append(element("div", "No saved tuning summaries are available.", "unavailable")); return; }
  state.tuning.forEach(item => {
    const card = element("article", undefined, "card"); card.append(element("h3", item.run_id));
    card.append(element("p", `Status: ${item.summary.status || "unknown"}`, "meta"));
    const selections = item.summary.selections && typeof item.summary.selections === "object" ? item.summary.selections : {};
    Object.entries(selections).forEach(([method, strength]) => card.append(element("p", `Development-selected ${method} strength: ${strength}`, "eyebrow")));
    const aggregates = Array.isArray(item.summary.aggregates) ? item.summary.aggregates : [];
    aggregates.forEach(row => card.append(element("p", `${row.method} · strength ${row.strength}: balanced ${format(row.mean_balanced_score)}, bias avoidance ${format(row.mean_bias_avoidance)}, quality ${format(row.mean_response_quality)} (n=${row.n})`)));
    target.append(card);
  });
}
function format(value) { return typeof value === "number" ? value.toFixed(3) : "unavailable"; }

async function compareRuns() {
  const message = byId("generation-message"); const target = byId("generation-content"); clear(target); message.textContent = "Loading comparison…";
  try {
    const result = await api("/api/generations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_keys: [byId("run-a").value, byId("run-b").value] }) });
    message.textContent = result.warnings.length ? result.warnings.join(" ") : "Runs pass saved-generation compatibility checks. Prompts remain unavailable until benchmark reconstruction is verified.";
    result.examples.forEach(example => {
      const article = element("article", undefined, "generation"); const heading = element("h3", example.example_id); article.append(heading);
      Object.entries(example.outputs).forEach(([method, output]) => { const lane = element("div"); lane.append(element("strong", method)); lane.append(element("div", output, "output")); article.append(lane); });
      target.append(article);
    });
  } catch (error) { message.textContent = error.message; message.classList.add("error"); }
}

async function renderControls() {
  const target = byId("controls-content"); clear(target);
  try {
    const result = await api("/api/controls");
    if (result.status !== "available") { target.append(element("div", result.reason, "unavailable")); return; }
    result.records.forEach(record => { const card = element("article", undefined, "card"); card.append(element("h3", `${record.observed.method} · ${record.metric.name}`)); card.append(element("p", `${record.family}. ${record.summary.count} controls. Empirical percentile: ${format(record.summary.percentile)}.`)); card.append(element("p", `Observed run: ${record.observed.run_id}`, "meta")); target.append(card); });
  } catch (error) { target.append(element("div", error.message, "error")); }
}

async function renderRepresentation() {
  const target = byId("representation-content"); clear(target);
  try {
    const result = await api("/api/representation");
    if (result.status !== "available") { target.append(element("div", result.reason, "unavailable")); return; }
    result.bundles.forEach(bundle => { const card = element("article", undefined, "card"); card.append(element("h3", `${bundle.config.model.name} · ${bundle.config.source_split}`)); const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"); renderScatter(svg, bundle.records); card.append(svg); card.append(element("p", `PCA explained variance: ${bundle.explained_variance_ratio.map(format).join(", ")}`, "meta")); target.append(card); });
  } catch (error) { target.append(element("div", error.message, "error")); }
}

function renderScatter(svg, rows) {
  svg.setAttribute("viewBox", "0 0 700 340"); const xs = rows.map(row => row.x); const ys = rows.map(row => row.y); const x0 = Math.min(...xs); const x1 = Math.max(...xs); const y0 = Math.min(...ys); const y1 = Math.max(...ys);
  rows.forEach(row => { const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle"); dot.setAttribute("cx", String(30 + 640 * (row.x - x0) / (x1 - x0 || 1))); dot.setAttribute("cy", String(310 - 280 * (row.y - y0) / (y1 - y0 || 1))); dot.setAttribute("r", "6"); dot.setAttribute("fill", row.state === "baseline" ? "#8b98a5" : "#55d6be"); dot.setAttribute("class", "dot"); const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = `${row.method} · ${row.example_id}`; dot.append(title); svg.append(dot); });
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
byId("compare-runs").addEventListener("click", compareRuns);
start();

