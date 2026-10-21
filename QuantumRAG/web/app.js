// QuantumRAG workbench front end. No framework, no build step.

const $ = (sel, root = document) => root.querySelector(sel);
const view = $("#view");
const tip = $("#tooltip");

const PIPES = {
  classical: { name: "Classical", color: "var(--c-classical)", desc: "Cosine similarity over 384-d embeddings" },
  quantum: { name: "Pure quantum", color: "var(--c-quantum)", desc: "Fidelity against every document" },
  hybrid: { name: "Hybrid", color: "var(--c-hybrid)", desc: "Classical shortlist, quantum re-rank" },
  pca_cosine: { name: "PCA control", color: "var(--c-control)", desc: "Cosine on the reduced features" },
};
const METHOD_NAMES = { kernel: "Kernel fidelity", swap: "SWAP test", iqae: "IQAE" };
const ENC_NAMES = { angle: "Angle", iqp: "IQP", amplitude: "Amplitude" };

const state = {
  settings: { encoding: "amplitude", n_qubits: 8, method: "kernel", shots: null, layers: 2,
              bandwidth: 1.0, fusion: 0, execution: "statevector", candidates: 50, k: 5,
              answerPipeline: "hybrid", provider: "extractive" },
  query: "",
  lastSearch: null,
  status: null,
  samples: [],
  benchMetric: "ndcg@10",
  concBandwidth: 1.0,
  corpus: { offset: 0, q: "" },
};

// ------------------------------------------------------------------ utils
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
const fmtMs = (v) => (v == null ? "n/a" : v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString());
const fmtBytes = (b) => (b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`);
const pct = (v) => `${(v * 100).toFixed(1)}%`;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    let msg = data?.detail ?? `Request failed with status ${res.status}`;
    if (Array.isArray(msg)) msg = msg.map((m) => `${(m.loc || []).slice(1).join(".")}: ${m.msg}`).join("; ");
    throw new Error(msg);
  }
  return data;
}

function quantumParams() {
  const s = state.settings;
  return { n_qubits: s.n_qubits, encoding: s.encoding, method: s.method, layers: s.layers,
           bandwidth: s.bandwidth, shots: s.shots, execution: s.execution, fusion: s.fusion };
}

function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };
function bindTips(root) {
  root.querySelectorAll("[data-tip]").forEach((el) => {
    el.addEventListener("mousemove", (e) => showTip(e, el.getAttribute("data-tip")));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("focus", (e) => { const r = el.getBoundingClientRect(); showTip({ clientX: r.right, clientY: r.top }, el.getAttribute("data-tip")); });
    el.addEventListener("blur", hideTip);
  });
}

// ------------------------------------------------------------------ theme
(function initTheme() {
  const btn = $("#theme-toggle");
  const current = () => document.documentElement.dataset.theme ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const label = () => { btn.textContent = current() === "dark" ? "Light theme" : "Dark theme"; };
  try { const saved = localStorage.getItem("qrag-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch { /* storage unavailable */ }
  label();
  btn.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("qrag-theme", next); } catch { /* ignore */ }
    label();
    route();
  });
})();

// ----------------------------------------------------------------- status
async function loadStatus() {
  const box = $("#rail-status");
  for (let attempt = 0; attempt < 120; attempt++) {
    try {
      const s = await api("/api/status");
      state.status = s;
      box.innerHTML = `
        <dl>
          <dt>Documents</dt><dd>${s.documents.toLocaleString()}</dd>
          <dt>Queries</dt><dd>${s.queries.toLocaleString()}</dd>
          <dt>Embedder</dt><dd>${s.embedder_backend === "minilm" ? "MiniLM-L6" : "LSA"}</dd>
          <dt>Index</dt><dd>${s.faiss ? "FAISS" : "NumPy"}</dd>
          <dt>PennyLane</dt><dd>${esc(s.pennylane)}</dd>
          <dt>Qiskit</dt><dd>${s.qiskit ? esc(s.qiskit) : "not installed"}</dd>
        </dl>
        ${s.embedder_backend !== "minilm" ? `<p class="note">Sentence-Transformers is not installed, so embeddings use the LSA fallback.</p>` : ""}`;
      return s;
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  box.innerHTML = `<p class="note">The server is not responding. Check the terminal running <span class="mono">python run.py</span>.</p>`;
  return null;
}

// ----------------------------------------------------------------- router
const routes = { retrieve: renderRetrieve, circuit: renderCircuit, benchmarks: renderBenchmarks, corpus: renderCorpus, method: renderMethod };
function route() {
  const name = (location.hash.replace(/^#\/?/, "") || "retrieve").split("?")[0];
  const fn = routes[name] || renderRetrieve;
  document.querySelectorAll(".rail nav a").forEach((a) => { if (a.dataset.route === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  hideTip();
  fn();
}
addEventListener("hashchange", () => { route(); view.focus({ preventScroll: true }); });

// --------------------------------------------------------- settings panel
function segmented(name, options, value) {
  return `<div class="seg" role="group" aria-label="${esc(name)}">${options.map(([v, l]) =>
    `<button type="button" data-set="${esc(name)}" data-val="${esc(v)}" aria-pressed="${String(v) === String(value)}">${esc(l)}</button>`).join("")}</div>`;
}
function qubitOptions() {
  const max = state.settings.encoding === "amplitude" ? 8 : 12;
  return Array.from({ length: max - 1 }, (_, i) => i + 2);
}
function settingsHTML({ showRetrieval = true } = {}) {
  const s = state.settings;
  const shotOpts = [["", "Exact"], ["256", "256"], ["1024", "1,024"], ["4096", "4,096"]];
  return `
  <div class="settings">
    <label class="field"><span>Encoding</span>${segmented("encoding", [["angle", "Angle"], ["iqp", "IQP"], ["amplitude", "Amplitude"]], s.encoding)}</label>
    <label class="field"><span>Qubits</span>
      <select data-set="n_qubits">${qubitOptions().map((n) => `<option value="${n}" ${n === s.n_qubits ? "selected" : ""}>${n}</option>`).join("")}</select></label>
    <label class="field"><span>Similarity estimator</span>${segmented("method", [["kernel", "Kernel fidelity"], ["swap", "SWAP test"], ["iqae", "IQAE"]], s.method)}</label>
    <label class="field"><span>Shots</span>
      <select data-set="shots" ${s.method === "iqae" ? "disabled" : ""}>${shotOpts.map(([v, l]) => `<option value="${v}" ${String(s.shots ?? "") === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
    ${showRetrieval ? `
    <label class="field"><span>Hybrid shortlist</span>
      <select data-set="candidates">${[10, 20, 50, 100, 200].map((n) => `<option value="${n}" ${n === s.candidates ? "selected" : ""}>${n} candidates</option>`).join("")}</select></label>
    <label class="field"><span>Results</span>
      <select data-set="k">${[3, 5, 10].map((n) => `<option value="${n}" ${n === s.k ? "selected" : ""}>Top ${n}</option>`).join("")}</select></label>` : ""}
  </div>
  <details class="more">
    <summary>More settings</summary>
    <div class="settings">
      <label class="field"><span>Feature-map layers</span>
        <select data-set="layers" ${s.encoding === "amplitude" ? "disabled" : ""}>${[1, 2, 3, 4].map((n) => `<option value="${n}" ${n === s.layers ? "selected" : ""}>${n}</option>`).join("")}</select></label>
      <label class="field"><span>Bandwidth</span><span class="range">
        <input type="range" min="0.1" max="2" step="0.05" value="${s.bandwidth}" data-set="bandwidth" ${s.encoding === "amplitude" ? "disabled" : ""}><output>${s.bandwidth.toFixed(2)}</output></span></label>
      ${showRetrieval ? `<label class="field"><span>Hybrid score fusion</span><span class="range">
        <input type="range" min="0" max="1" step="0.05" value="${s.fusion}" data-set="fusion"><output>${s.fusion.toFixed(2)}</output></span></label>` : ""}
      <label class="field"><span>Execution</span>${segmented("execution", [["statevector", "Cached statevectors"], ["circuit", "Full circuits"]], s.execution)}</label>
    </div>
    <p class="hint">${s.encoding === "amplitude" ? "Amplitude encoding stores 2^N features in N qubits, so layers and bandwidth do not apply." :
      "Bandwidth below 1 shrinks rotation angles, which counters kernel concentration at higher qubit counts."}
      Fusion blends the hybrid's quantum fidelity with the classical cosine (0 = pure quantum re-rank).
      Full-circuit execution simulates one circuit per document pair; use it with the hybrid pipeline.</p>
  </details>`;
}
function bindSettings(root, onChange) {
  root.querySelectorAll("[data-set]").forEach((el) => {
    const key = el.dataset.set;
    const apply = (raw) => {
      const s = state.settings;
      if (["n_qubits", "layers", "candidates", "k"].includes(key)) s[key] = parseInt(raw, 10);
      else if (["bandwidth", "fusion"].includes(key)) s[key] = parseFloat(raw);
      else if (key === "shots") s.shots = raw ? parseInt(raw, 10) : null;
      else s[key] = raw;
      if (s.encoding === "amplitude" && s.n_qubits > 8) s.n_qubits = 8;
      onChange(key);
    };
    if (el.tagName === "BUTTON") el.addEventListener("click", () => apply(el.dataset.val));
    else if (el.type === "range") {
      el.addEventListener("input", () => { el.nextElementSibling.textContent = parseFloat(el.value).toFixed(2); });
      el.addEventListener("change", () => apply(el.value));
    } else el.addEventListener("change", () => apply(el.value));
  });
}

// =============================================================== RETRIEVE
async function renderRetrieve() {
  view.innerHTML = `
    <header class="page-head">
      <h1>Compare retrieval pipelines</h1>
      <p>Ask a question about the corpus, or <a href="#/corpus">upload your own PDF or text files</a>. The same query runs through classical cosine search,
      pure quantum similarity search and the hybrid re-ranker, so you can see where their rankings agree and where they part.</p>
    </header>
    <section class="panel" id="controls">
      <form class="querybar" id="qform">
        <label class="visually-hidden" for="q">Question</label>
        <input id="q" name="q" autocomplete="off" placeholder="${state.status?.corpus === "custom" ? "Ask a question about your uploaded files" : "For example: What did Tesla build in Colorado Springs?"}" value="${esc(state.query)}">
        <button class="btn" type="submit" id="run">Run search</button>
      </form>
      <div id="settings">${settingsHTML()}</div>
    </section>
    <div class="samples" id="samples" aria-label="Sample questions from the benchmark"></div>
    <div id="out"></div>`;

  const form = $("#qform");
  form.addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  const rebind = () => {
    const open = $("#settings details")?.open;
    $("#settings").innerHTML = settingsHTML();
    if (open) $("#settings details").open = true;
    bindSettings($("#settings"), () => { rebind(); if (state.query) runSearch(); });
  };
  rebind();

  if (!state.samples.length && state.status?.corpus !== "custom") {
    try { state.samples = (await api("/api/samples?n=6")).samples; } catch { /* server warming up */ }
  }
  if (state.status?.corpus === "custom") {
    $("#samples").outerHTML = `<p class="hint" style="margin-top:12px">Searching ${state.status.documents.toLocaleString()} passages from your uploaded files. <a href="#/corpus">Manage files</a></p>`;
    if (state.lastSearch) drawResults(state.lastSearch);
    return;
  }
  $("#samples").innerHTML = state.samples.map((s, i) =>
    `<button type="button" data-i="${i}" title="Expected source: ${esc(s.relevant_title)}">${esc(s.query)}</button>`).join("");
  $("#samples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    $("#q").value = state.samples[+b.dataset.i].query; runSearch();
  }));
  if (state.lastSearch) drawResults(state.lastSearch);
}

let searchSeq = 0;
async function runSearch() {
  const q = $("#q").value.trim();
  if (!q) { $("#q").focus(); return; }
  state.query = q;
  const seq = ++searchSeq;
  const btn = $("#run");
  btn.disabled = true;
  $("#out").innerHTML = `<p class="loading"><span class="spinner"></span>Running the three pipelines.</p>`;
  try {
    const s = state.settings;
    const res = await api("/api/search", { query: q, pipelines: ["classical", "quantum", "hybrid"],
      k: s.k, candidates: s.candidates, ...quantumParams() });
    if (seq !== searchSeq) return;
    state.lastSearch = res;
    drawResults(res);
    runAnswer();
  } catch (err) {
    if (seq === searchSeq) $("#out").innerHTML = `<div class="error">${esc(err.message)}</div>`;
  } finally {
    if (btn.isConnected) btn.disabled = false;
  }
}

function drawResults(res) {
  const cfg = res.config;
  const gold = new Set(res.relevant || []);
  const qs = res.quantum_space;
  const encLabel = `${ENC_NAMES[cfg.encoding]} encoding, ${cfg.n_qubits} qubits, ${METHOD_NAMES[cfg.method]}${cfg.shots ? `, ${cfg.shots.toLocaleString()} shots` : cfg.method === "iqae" ? "" : ", exact"}`;
  const col = (p) => {
    const r = res.results[p];
    const t = r.timings_ms;
    const parts = p === "classical" ? [] :
      p === "quantum" ? [["Encode query", t.encode_query], ["Fidelity", t.quantum_similarity]] :
      [["Shortlist", t.coarse_filter], ["Quantum re-rank", t.quantum_rerank]];
    const st = r.stats || {};
    const desc = p === "hybrid" ? `Top ${Math.min(res.candidates, state.status?.documents ?? res.candidates)} by cosine, re-ranked by fidelity${cfg.fusion > 0 ? ` (fusion ${cfg.fusion})` : ""}`
      : p === "quantum" ? `Fidelity against all ${state.status?.documents?.toLocaleString() ?? ""} documents` : PIPES[p].desc;
    const maxScore = Math.max(...r.items.map((i) => i.score), 1e-9);
    return `
    <section class="panel col" data-p="${p}" aria-label="${PIPES[p].name} results">
      <div class="col-head">
        <div class="row"><h2>${PIPES[p].name}</h2>
          <span class="lat" title="Search time, excluding the shared query embedding">${fmtMs(t.search)}<small>ms</small></span></div>
        <p>${esc(desc)}</p>
        ${parts.length || st.circuits_executed ? `<div class="breakdown">
          ${parts.map(([l, v]) => `<span>${l} ${fmtMs(v)} ms</span>`).join("")}
          ${st.circuits_executed ? `<span>${st.circuits_executed.toLocaleString()} circuits on ${st.qubits_per_circuit} qubits</span>` : ""}
          ${st.shots_used ? `<span>${st.shots_used.toLocaleString()} shots</span>` : ""}
          ${st.oracle_queries ? `<span>${st.oracle_queries.toLocaleString()} Grover queries</span>` : ""}
        </div>` : ""}
      </div>
      <ol class="results">${r.items.map((it) => {
        const isGold = gold.has(it.id);
        let shift = "";
        if (p !== "classical") {
          const cr = it.classical_rank;
          if (cr == null) shift = `<span class="shift down">not in classical top ${res.k}</span>`;
          else if (cr !== it.rank) shift = `<span class="shift ${cr > it.rank ? "up" : "down"}">classical rank ${cr}</span>`;
          else shift = `<span class="shift">same as classical</span>`;
        }
        return `<li class="result ${isGold ? "is-gold" : ""}">
          <span class="rk">${it.rank}</span>
          <div>
            <h3>${esc(it.title)}</h3>
            <div class="meta">
              <span class="scorebar" data-tip="${p === "classical" ? "Cosine similarity" : "Estimated fidelity"} ${fmt(it.score, 4)}${it.cosine != null && p !== "classical" ? `<br>Cosine ${fmt(it.cosine, 4)}` : ""}" tabindex="0">
                <i><b style="width:${Math.max(2, (it.score / maxScore) * 100)}%"></b></i><span>${fmt(it.score, 3)}</span></span>
              ${isGold ? `<span class="gold-tag">Expected source</span>` : ""}
              ${shift}
            </div>
            <p class="snippet" hidden>${esc(it.text)}</p>
            <button type="button" class="linklike toggle">Show passage</button>
          </div>
        </li>`;
      }).join("")}</ol>
    </section>`;
  };
  $("#out").innerHTML = `
    <div class="summary-line">
      <span>Query embedded in <strong>${fmtMs(res.embedding_ms)} ms</strong></span>
      <span>${esc(encLabel)}</span>
      ${qs ? `<span>PCA keeps <strong>${pct(qs.explained_variance)}</strong> of embedding variance in ${qs.feature_dim} features</span>` : ""}
      ${gold.size ? `<span>Highlighted rows are the benchmark's expected source</span>` : ""}
    </div>
    <div class="columns">${["classical", "quantum", "hybrid"].map(col).join("")}</div>
    <section class="panel answer" id="answer"></section>`;
  $("#out").querySelectorAll(".toggle").forEach((b) => b.addEventListener("click", () => {
    const sn = b.previousElementSibling; sn.hidden = !sn.hidden; b.textContent = sn.hidden ? "Show passage" : "Hide passage";
  }));
  bindTips($("#out"));
  drawAnswerShell();
}

function drawAnswerShell(content = `<p class="loading"><span class="spinner"></span>Writing an answer from the retrieved passages.</p>`) {
  const s = state.settings;
  const box = $("#answer");
  if (!box) return;
  box.innerHTML = `
    <div class="answer-head">
      <div><h2>Answer</h2><p class="hint">Generated only from the passages the chosen pipeline retrieved.</p></div>
      <div class="controls">
        <label class="field"><span>Context from</span>${segmented("answerPipeline", [["classical", "Classical"], ["quantum", "Pure quantum"], ["hybrid", "Hybrid"]], s.answerPipeline)}</label>
        <label class="field"><span>Generator</span><select data-set="provider">
          ${[["extractive", "Extractive (offline)"], ["ollama", "Ollama, local LLM"], ["openai", "OpenAI-compatible API"]].map(([v, l]) => `<option value="${v}" ${v === s.provider ? "selected" : ""}>${l}</option>`).join("")}
        </select></label>
      </div>
    </div>
    <div id="answer-body">${content}</div>`;
  bindSettings(box, () => { drawAnswerShell(); runAnswer(); });
}

let answerSeq = 0;
async function runAnswer() {
  const s = state.settings;
  const seq = ++answerSeq;
  try {
    const a = await api("/api/answer", { query: state.query, pipeline: s.answerPipeline, provider: s.provider,
      k: s.k, candidates: s.candidates, ...quantumParams() });
    if (seq !== answerSeq || !$("#answer-body")) return;
    const text = esc(a.answer).replace(/\[(\d+)\]/g, (_, n) => `<cite title="Source ${n}">${n}</cite>`);
    $("#answer-body").innerHTML = `
      <div class="answer-body">${text}</div>
      ${a.note ? `<p class="notice">${esc(a.note)}</p>` : ""}
      <ol class="sources">${a.sources.map((src) => `<li><span class="num">[${src.n}]</span><span>${esc(src.title)}</span><span class="muted num">${fmt(src.score, 3)}</span></li>`).join("")}</ol>
      <p class="hint" style="margin-top:10px">${esc(a.model)}${a.confidence != null ? `, confidence ${fmt(a.confidence, 2)}` : ""}, generated in ${fmtMs(a.latency_ms)} ms</p>`;
  } catch (err) {
    if ($("#answer-body")) $("#answer-body").innerHTML = `<div class="error">${esc(err.message)}</div>`;
  }
}

// ================================================================ CIRCUIT
async function renderCircuit() {
  view.innerHTML = `
    <header class="page-head">
      <h1>Similarity circuit</h1>
      <p>The exact gate sequence PennyLane simulates to compare your query with its closest document.
      Change the encoding or estimator to see how the circuit, its width and its depth change.</p>
    </header>
    <section class="panel" id="controls">
      <form class="querybar" id="cform">
        <label class="visually-hidden" for="cq">Question</label>
        <input id="cq" placeholder="Question to encode (leave empty to compare two corpus documents)" value="${esc(state.query)}">
        <button class="btn" type="submit">Draw circuit</button>
      </form>
      <div id="settings"></div>
    </section>
    <div id="cout" style="margin-top:16px"></div>`;
  const rebind = () => {
    const open = $("#settings details")?.open;
    $("#settings").innerHTML = settingsHTML({ showRetrieval: false });
    if (open) $("#settings details").open = true;
    bindSettings($("#settings"), () => { rebind(); drawCircuit(); });
  };
  rebind();
  $("#cform").addEventListener("submit", (e) => { e.preventDefault(); state.query = $("#cq").value.trim(); drawCircuit(); });
  drawCircuit();
}

let circuitSeq = 0;
async function drawCircuit() {
  const out = $("#cout");
  const seq = ++circuitSeq;
  out.innerHTML = `<p class="loading"><span class="spinner"></span>Building the circuit.</p>`;
  try {
    const c = await api("/api/circuit", { ...quantumParams(), query: $("#cq").value.trim() || null });
    if (seq !== circuitSeq) return;
    const width = c.method === "swap" ? `${c.n_wires} <small>(2N + 1)</small>` : c.n_wires;
    out.innerHTML = `
      <section class="panel">
        <dl class="facts">
          <div class="fact"><dt>Qubits</dt><dd class="num">${width}</dd></div>
          <div class="fact"><dt>Depth</dt><dd class="num">${c.depth}</dd></div>
          <div class="fact"><dt>Gates</dt><dd class="num">${c.gate_count}</dd></div>
          <div class="fact"><dt>Multi-qubit gates</dt><dd class="num">${c.two_qubit_gates}</dd></div>
          <div class="fact"><dt>Fidelity with document</dt><dd class="num">${fmt(c.fidelity, 4)}</dd></div>
        </dl>
        <div class="circuit-scroll" id="circ"></div>
        <div class="group-legend" id="glegend"></div>
      </section>
      <div class="two-up" style="margin-top:16px">
        <section class="panel chart"><h2>Query state</h2>
          <p class="sub">Probability of each computational basis state after encoding the query${c.query_probabilities.length < 2 ** (c.method === "swap" ? (c.n_wires - 1) / 2 : c.n_wires) ? ` (first ${c.query_probabilities.length} states shown)` : ""}.</p>
          <div id="probs"></div></section>
        <section class="panel card-pad explain" id="cexplain"></section>
      </div>`;
    drawCircuitSVG($("#circ"), c);
    drawProbChart($("#probs"), c.query_probabilities);
    $("#cexplain").innerHTML = circuitExplanation(c);
  } catch (err) {
    if (seq === circuitSeq) out.innerHTML = `<div class="error">${esc(err.message)}</div>`;
  }
}

function groupColor(name) {
  if (name.includes(" and ")) return "var(--ink-2)";
  if (name.startsWith("U(query)")) return "var(--c-quantum)";
  if (name.startsWith("U(doc)")) return "var(--c-hybrid)";
  return "var(--s1)";
}

function drawCircuitSVG(host, c) {
  const colW = 50, rowH = 46, left = 58, top = 40, right = 54;
  const W = left + c.depth * colW + right;
  const H = top + c.n_wires * rowH + 6;
  const y = (w) => top + w * rowH + rowH / 2;
  const x = (col) => left + col * colW + colW / 2;
  const parts = [];
  // group bands
  c.groups.forEach((g) => {
    const x0 = left + g.start * colW + 3, x1 = left + (g.end + 1) * colW - 3;
    parts.push(`<rect x="${x0}" y="8" width="${Math.max(4, x1 - x0)}" height="4" rx="2" fill="${groupColor(g.name)}"/>`);
    if (x1 - x0 > 60) parts.push(`<text x="${x0}" y="27" font-size="12" fill="var(--ink-2)">${esc(g.name)}</text>`);
  });
  // wires
  for (let w = 0; w < c.n_wires; w++) {
    parts.push(`<text x="${left - 14}" y="${y(w) + 4}" text-anchor="end" font-size="12.5" fill="var(--muted)">${esc(c.wire_labels[w])}</text>`);
    parts.push(`<line x1="${left - 6}" x2="${W - right + 20}" y1="${y(w)}" y2="${y(w)}" stroke="var(--rule-strong)" stroke-width="1.2"/>`);
  }
  // measurement
  c.measured.forEach((w) => {
    const mx = W - right + 22, my = y(w);
    parts.push(`<g><rect x="${mx}" y="${my - 12}" width="28" height="24" rx="3" fill="var(--surface)" stroke="var(--ink-2)"/>
      <path d="M${mx + 6} ${my + 6} A8 8 0 0 1 ${mx + 22} ${my + 6}" fill="none" stroke="var(--ink-2)"/>
      <line x1="${mx + 14}" y1="${my + 6}" x2="${mx + 20}" y2="${my - 6}" stroke="var(--ink-2)"/></g>`);
  });
  const box = (cx, cy, label, color, p, tipText) => {
    const wB = Math.max(34, label.length * 7.4 + 12);
    return `<g tabindex="0" data-tip="${esc(tipText)}"><rect x="${cx - wB / 2}" y="${cy - 12}" width="${wB}" height="24" rx="4" fill="var(--surface)" stroke="${color}" stroke-width="1.6"/>
      <text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--ink)">${esc(label)}</text>
      ${p != null ? `<text class="param" x="${cx}" y="${cy + 22}" text-anchor="middle" font-size="9.5" fill="var(--muted)">${p}</text>` : ""}</g>`;
  };
  c.ops.forEach((op) => {
    const cx = x(op.col);
    const color = groupColor(op.group);
    const p = op.params.length ? op.params[0].toFixed(2) : null;
    const tipText = `<b>${op.gate}</b> on ${op.wires.map((w) => c.wire_labels[w]).join(", ")}${op.params.length ? `<br>angle ${op.params[0].toFixed(4)} rad` : ""}<br>${op.group}`;
    const ws = op.wires;
    if (op.gate === "CNOT") {
      parts.push(`<g tabindex="0" data-tip="${esc(tipText)}"><line x1="${cx}" x2="${cx}" y1="${y(ws[0])}" y2="${y(ws[1])}" stroke="${color}" stroke-width="1.6"/>
        <circle cx="${cx}" cy="${y(ws[0])}" r="4.5" fill="${color}"/>
        <circle cx="${cx}" cy="${y(ws[1])}" r="10" fill="var(--surface)" stroke="${color}" stroke-width="1.6"/>
        <line x1="${cx - 10}" x2="${cx + 10}" y1="${y(ws[1])}" y2="${y(ws[1])}" stroke="${color}" stroke-width="1.6"/>
        <line x1="${cx}" x2="${cx}" y1="${y(ws[1]) - 10}" y2="${y(ws[1]) + 10}" stroke="${color}" stroke-width="1.6"/></g>`);
    } else if (op.gate === "CSWAP") {
      const xs = (yy) => `<path d="M${cx - 6} ${yy - 6} L${cx + 6} ${yy + 6} M${cx + 6} ${yy - 6} L${cx - 6} ${yy + 6}" stroke="${color}" stroke-width="2"/>`;
      parts.push(`<g tabindex="0" data-tip="${esc(tipText)}"><line x1="${cx}" x2="${cx}" y1="${y(Math.min(...ws))}" y2="${y(Math.max(...ws))}" stroke="${color}" stroke-width="1.6"/>
        <circle cx="${cx}" cy="${y(ws[0])}" r="4.5" fill="${color}"/>${xs(y(ws[1]))}${xs(y(ws[2]))}</g>`);
    } else if (ws.length === 2 && op.gate.startsWith("ZZ")) {
      parts.push(`<line x1="${cx}" x2="${cx}" y1="${y(ws[0])}" y2="${y(ws[1])}" stroke="${color}" stroke-width="1.6"/>`);
      const [top, bot] = y(ws[0]) < y(ws[1]) ? [ws[0], ws[1]] : [ws[1], ws[0]];
      parts.push(box(cx, y(top), op.gate, color, null, tipText));
      parts.push(box(cx, y(bot), op.gate, color, p, tipText));
    } else if (ws.length > 1) {
      const y0 = y(Math.min(...ws)) - 14, y1 = y(Math.max(...ws)) + 14;
      parts.push(`<g tabindex="0" data-tip="${esc(tipText)}"><rect x="${cx - 21}" y="${y0}" width="42" height="${y1 - y0}" rx="4" fill="var(--surface)" stroke="${color}" stroke-width="1.6"/>
        <text x="${cx}" y="${(y0 + y1) / 2}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--ink)" transform="rotate(-90 ${cx} ${(y0 + y1) / 2})">${esc(op.gate)}</text></g>`);
    } else {
      parts.push(box(cx, y(ws[0]), op.gate, color, p, tipText));
    }
  });
  host.innerHTML = `<svg class="circuit-svg" width="${W + 30}" height="${H}" viewBox="0 0 ${W + 30} ${H}" role="img" aria-label="Quantum circuit with ${c.gate_count} gates on ${c.n_wires} qubits">${parts.join("")}</svg>`;
  bindTips(host);
  const names = [...new Set(c.ops.map((o) => o.group))];
  $("#glegend").innerHTML = names.map((n) => `<span><i style="background:${groupColor(n)}"></i>${esc(n)}</span>`).join("") +
    `<span class="muted">Scroll sideways for long circuits. Hover a gate for its exact angle.</span>`;
}

function circuitExplanation(c) {
  const n = c.method === "swap" ? (c.n_wires - 1) / 2 : c.n_wires;
  const enc = {
    angle: `Angle encoding writes each PCA feature into one qubit as RY and RZ rotations, then entangles neighbours with a CNOT ring. ${n} qubits hold ${n} features.`,
    iqp: `IQP encoding puts every qubit in superposition with H, adds feature-dependent RZ phases and IsingZZ couplings proportional to products of neighbouring features, which makes the kernel non-linear in the input. ${n} qubits hold ${n} features.`,
    amplitude: `Amplitude encoding prepares a state whose ${2 ** n} amplitudes are the normalised PCA features, so ${n} qubits hold ${2 ** n} features. The fidelity then equals the squared cosine of the two reduced vectors.`,
  }[c.encoding];
  const est = {
    kernel: `The kernel circuit applies U(query) and then the inverse of U(doc). The probability of reading all zeros is exactly the fidelity |&lang;&psi;(doc)|&psi;(query)&rang;|&sup2;.`,
    swap: `The SWAP test prepares query and document on separate registers, then interferes them through an ancilla. The ancilla reads 0 with probability (1 + F) / 2, so F = 2P(0) &minus; 1. It needs 2N + 1 qubits but never inverts a circuit.`,
    iqae: `IQAE treats the kernel circuit as the state preparation A and the all-zeros outcome as the good state. Measuring Q<sup>k</sup>A for growing powers k narrows the estimate of F with about 1/&epsilon; circuit calls instead of 1/&epsilon;&sup2; shots. The last block shows one Grover operator Q.`,
  }[c.method];
  return `<h2>How to read it</h2><p>${enc}</p><p>${est}</p>
    <p class="muted">Compared document: ${esc(c.document.title)}. Feature values shown in the gate angles are the PCA components scaled to the rotation range.</p>`;
}

function drawProbChart(host, probs) {
  const n = probs.length, W = 520, H = 190, padL = 40, padB = 26, padT = 8;
  const max = Math.max(...probs, 1e-9);
  const yMax = Math.min(1, Math.ceil(max * 10) / 10 || 0.1);
  const bw = (W - padL) / n;
  const barW = Math.min(18, Math.max(2, bw - 2));
  const ticks = [0, yMax / 2, yMax];
  const yy = (v) => padT + (H - padT - padB) * (1 - v / yMax);
  const bits = Math.max(1, Math.round(Math.log2(n)));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Basis-state probabilities of the encoded query">`;
  s += `<g class="grid">${ticks.map((t) => `<line x1="${padL}" x2="${W}" y1="${yy(t)}" y2="${yy(t)}"/><text x="${padL - 6}" y="${yy(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`).join("")}</g>`;
  probs.forEach((p, i) => {
    const x0 = padL + i * bw + (bw - barW) / 2, h = Math.max(0, H - padB - yy(p));
    const label = i.toString(2).padStart(bits, "0");
    s += `<g data-tip="|${label}&rang;  probability ${p.toFixed(4)}" tabindex="-1"><rect x="${padL + i * bw}" y="${padT}" width="${bw}" height="${H - padT - padB}" fill="transparent"/>
      <path d="${barPath(x0, H - padB, barW, h)}" fill="var(--c-quantum)"/></g>`;
  });
  const step = Math.max(1, Math.ceil(n / 8));
  for (let i = 0; i < n; i += step) s += `<text x="${padL + i * bw + bw / 2}" y="${H - 8}" text-anchor="middle">${i}</text>`;
  s += `</svg>`;
  host.innerHTML = s;
  bindTips(host);
}

// vertical bar with 4px rounded data end, square baseline
function barPath(x, base, w, h) {
  const r = Math.min(4, w / 2, h);
  if (h <= 0) return "";
  return `M${x} ${base} V${base - h + r} Q${x} ${base - h} ${x + r} ${base - h} H${x + w - r} Q${x + w} ${base - h} ${x + w} ${base - h + r} V${base} Z`;
}
function hbarPath(x, yTop, len, t) {
  const r = Math.min(4, t / 2, len);
  if (len <= 0) return "";
  return `M${x} ${yTop} H${x + len - r} Q${x + len} ${yTop} ${x + len} ${yTop + r} V${yTop + t - r} Q${x + len} ${yTop + t} ${x + len - r} ${yTop + t} H${x} Z`;
}

// ============================================================= BENCHMARKS
let benchPoll = null;
async function renderBenchmarks() {
  view.innerHTML = `
    <header class="page-head">
      <h1>Benchmarks</h1>
      <p>Every pipeline evaluated on the same queries and corpus: retrieval quality, latency on the simulator,
      memory, and the analyses that explain the gaps.</p>
    </header>
    <div id="bench"><p class="loading"><span class="spinner"></span>Loading the latest results.</p></div>`;
  let data;
  try { data = await api("/api/benchmark"); } catch (err) { $("#bench").innerHTML = `<div class="error">${esc(err.message)}</div>`; return; }
  const runner = `
    <section class="panel card-pad" id="runner">
      <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;justify-content:space-between">
        <div><h2>${data.available ? "Run the suite again" : "No results yet"}</h2>
          <p class="hint">${data.available ? "Results are saved to the results folder with a timestamp." : "Run the benchmark to fill this page. 200 queries takes a few minutes on a laptop."}</p></div>
        <div style="display:flex;gap:10px;align-items:flex-end">
          <label class="field"><span>Queries</span><select id="bn">${[100, 200, 300, 500, 1000].map((n) => `<option ${n === 300 ? "selected" : ""}>${n}</option>`).join("")}</select></label>
          <button class="btn" id="brun" type="button">Run benchmark</button>
        </div>
      </div>
      <div id="bprog"></div>
    </section>`;
  if (!data.available) { $("#bench").innerHTML = runner; bindRunner(); return; }

  const runs = data.runs;
  const classical = runs.find((r) => r.pipeline === "classical");
  const env = data.environment;
  const metricCols = ["recall@1", "recall@5", "recall@10", "ndcg@10", "mrr@10"];
  const best = Object.fromEntries(metricCols.map((m) => [m, Math.max(...runs.map((r) => r.metrics[m] ?? 0))]));
  $("#bench").innerHTML = `
    <div class="bench-meta">
      <span><b>${esc(data.dataset)}</b></span>
      <span>${data.documents.toLocaleString()} documents, ${data.queries_evaluated.toLocaleString()} queries</span>
      <span>${esc(env.embedder)}</span>
      <span>PennyLane ${esc(env.pennylane)} on ${esc(env.device)}</span>
      <span>Run ${new Date(data.created).toLocaleString()}</span>
    </div>
    <section class="panel">
      <div class="table-wrap"><table class="data">
        <thead><tr><th>Configuration</th>${metricCols.map((m) => `<th>${m.replace("recall", "Recall").replace("ndcg", "NDCG").replace("mrr", "MRR")}</th>`).join("")}
          <th>Mean latency</th><th>p95 latency</th><th>Index memory</th><th>PCA variance kept</th></tr></thead>
        <tbody>${runs.map((r) => `<tr>
          <td><span class="dot" style="background:${PIPES[r.pipeline].color}"></span>${esc(r.label)}</td>
          ${metricCols.map((m) => `<td class="${r.metrics[m] === best[m] ? "best" : ""}">${fmt(r.metrics[m], 3)}</td>`).join("")}
          <td>${fmtMs(r.latency_ms.mean)} ms</td><td>${fmtMs(r.latency_ms.p95)} ms</td>
          <td>${fmtBytes(r.memory.index_bytes)}</td><td>${r.pipeline === "classical" ? "all" : pct(r.explained_variance)}</td></tr>`).join("")}</tbody>
      </table></div>
    </section>
    <section class="panel findings"><h2>What the numbers say</h2><ol>${findings(data).map((f) => `<li>${f}</li>`).join("")}</ol></section>
    <div class="chart-grid">
      <section class="panel chart" id="ch-quality"></section>
      <section class="panel chart" id="ch-latency"></section>
      <section class="panel chart" id="ch-variance"></section>
      <section class="panel chart" id="ch-conc"></section>
      <section class="panel chart" id="ch-est"></section>
      <section class="panel chart" id="ch-mem"></section>
    </div>
    <div style="margin-top:16px">${runner}</div>`;
  drawQualityChart(runs);
  drawLatencyChart(runs);
  drawMemoryChart(runs);
  if (data.analysis?.explained_variance) drawVarianceChart(data.analysis.explained_variance);
  if (data.analysis?.kernel_concentration) drawConcentration(data.analysis.kernel_concentration);
  if (data.analysis?.estimator_efficiency) drawEstimator(data.analysis.estimator_efficiency);
  bindRunner();
}

function bindRunner() {
  const btn = $("#brun");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await api("/api/benchmark/run", { n_queries: parseInt($("#bn").value, 10), analysis: true });
      pollBench();
    } catch (err) { $("#bprog").innerHTML = `<div class="error">${esc(err.message)}</div>`; btn.disabled = false; }
  });
  api("/api/benchmark/progress").then((p) => { if (p.running) { btn.disabled = true; pollBench(); } }).catch(() => {});
}
function pollBench() {
  clearInterval(benchPoll);
  benchPoll = setInterval(async () => {
    let p; try { p = await api("/api/benchmark/progress"); } catch { return; }
    const box = $("#bprog");
    if (!box) { clearInterval(benchPoll); return; }
    if (p.error) { clearInterval(benchPoll); box.innerHTML = `<div class="error">${esc(p.error)}</div>`; return; }
    const frac = p.total ? p.step / p.total : 0;
    box.innerHTML = `<div class="progress"><b style="width:${frac * 100}%"></b></div><p class="hint">Step ${Math.min(p.step + 1, p.total)} of ${p.total}: ${esc(p.label)}</p>`;
    if (!p.running) { clearInterval(benchPoll); renderBenchmarks(); }
  }, 1000);
}

function findings(d) {
  const runs = d.runs, out = [];
  const byLabel = (pred) => runs.filter(pred);
  const cl = runs.find((r) => r.pipeline === "classical");
  const nd = (r) => r.metrics["ndcg@10"];
  const q = byLabel((r) => r.pipeline === "quantum");
  const h = byLabel((r) => r.pipeline === "hybrid");
  if (cl && q.length) {
    const bestQ = q.reduce((a, b) => (nd(b) > nd(a) ? b : a));
    const diff = nd(bestQ) - nd(cl);
    out.push(`The best pure quantum configuration, ${esc(bestQ.label)}, reaches NDCG@10 ${fmt(nd(bestQ))} against ${fmt(nd(cl))} for classical cosine (${diff >= 0 ? "+" : ""}${fmt(diff)}).`);
  }
  const ctl8 = runs.find((r) => r.pipeline === "pca_cosine" && r.config?.n_qubits === 8 && r.config?.encoding !== "amplitude");
  const ang8 = runs.find((r) => r.pipeline === "quantum" && r.config?.encoding === "angle" && r.config?.n_qubits === 8);
  if (ctl8 && ang8 && cl) {
    const lostToPca = nd(cl) - nd(ctl8), lostToKernel = nd(ctl8) - nd(ang8);
    out.push(`Dimensionality reduction explains most of the gap for angle encoding: cosine on the same 8 PCA features already drops NDCG@10 by ${fmt(lostToPca)}, while the quantum kernel on top of those same features ${lostToKernel >= 0 ? "lowers it by a further" : "raises it by"} ${fmt(Math.abs(lostToKernel))}.`);
  }
  const amp8 = runs.find((r) => r.pipeline === "quantum" && r.config?.encoding === "amplitude" && r.config?.n_qubits === 8);
  if (amp8) out.push(`Amplitude encoding packs 256 features into 8 qubits and keeps ${pct(amp8.explained_variance)} of the embedding variance, against ${ang8 ? pct(ang8.explained_variance) : "far less"} for 8 angle-encoded features.`);
  const qAmp = amp8, hAmp = runs.find((r) => r.pipeline === "hybrid" && r.config?.encoding === "amplitude" && r.config?.method === "kernel");
  if (qAmp && hAmp) out.push(`The hybrid pipeline limits quantum work to ${d.candidates} candidates per query instead of all ${d.documents.toLocaleString()} documents, with NDCG@10 ${fmt(nd(hAmp))} versus ${fmt(nd(qAmp))} for the pure quantum run and a mean latency of ${fmtMs(hAmp.latency_ms.mean)} ms.`);
  const fus = runs.find((r) => r.config?.fusion > 0), hAng = runs.find((r) => r.pipeline === "hybrid" && r.config?.encoding === "angle" && !(r.config?.fusion > 0));
  if (fus && hAng) out.push(`Blending fidelity with cosine (fusion ${fus.config.fusion}) lifts the angle-encoded hybrid from ${fmt(nd(hAng))} to ${fmt(nd(fus))} NDCG@10.`);
  if (cl && q.length) {
    const slow = q.reduce((a, b) => (b.latency_ms.mean > a.latency_ms.mean ? b : a));
    out.push(`Classical search takes ${fmtMs(cl.latency_ms.mean)} ms per query; the slowest quantum configuration takes ${fmtMs(slow.latency_ms.mean)} ms. This is simulator overhead on a classical CPU, not a statement about quantum hardware.`);
  }
  const conc = d.analysis?.kernel_concentration;
  if (conc && conc.some((r) => r.source)) {
    const series = (src, enc, bw) => conc.filter((r) => r.source === src && r.encoding === enc && r.bandwidth === bw).sort((a, b) => a.n_qubits - b.n_qubits);
    const rnd = series("random", "iqp", 1.0), real = series("corpus", "iqp", 1.0), hot = series("corpus", "iqp", 2.0);
    if (rnd.length > 1 && real.length > 1) {
      const f = (r) => r.variance.toExponential(1), first = (a) => a[0], last = (a) => a[a.length - 1];
      out.push(`With random inputs the IQP kernel concentrates exponentially: fidelity variance goes from ${f(first(rnd))} at ${first(rnd).n_qubits} qubits to ${f(last(rnd))} at ${last(rnd).n_qubits}. On the corpus's PCA features it stays at ${f(last(real))} at ${last(real).n_qubits} qubits, because the features are correlated rather than random${hot.length > 1 ? `; raising the bandwidth to 2 brings it down to ${f(last(hot))}` : ""}. Bandwidth is therefore a real tuning knob for larger qubit counts.`);
    }
  }
  const est = d.analysis?.estimator_efficiency;
  if (est) {
    const i = est.iqae.find((r) => r.epsilon === 0.001) || est.iqae[est.iqae.length - 1];
    const target = i.mean_abs_error;
    const s = est.sampling.find((r) => r.mean_abs_error <= target);
    out.push(`IQAE reaches a mean error of ${target.toExponential(1)} with about ${Math.round(i.oracle_calls).toLocaleString()} circuit calls; plain sampling ${s ? `needs about ${s.oracle_calls.toLocaleString()} shots for the same error` : `does not reach that error even with ${est.sampling[est.sampling.length - 1].oracle_calls.toLocaleString()} shots`}.`);
  }
  return out;
}

// ---------------------------------------------------------- chart helpers
function niceMax(v) {
  if (v <= 0) return 1;
  const e = 10 ** Math.floor(Math.log10(v));
  const m = v / e;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * e;
}
function logTicks(lo, hi) {
  const out = [];
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) out.push(10 ** p);
  return out;
}
const supNum = (v) => {
  if (v >= 1 && v < 10000) return v.toLocaleString();
  const p = Math.round(Math.log10(v));
  return `1e${p}`;
};
function pipelineLegend(runs) {
  const seen = [...new Set(runs.map((r) => r.pipeline))];
  return `<div class="legend">${seen.map((p) => `<span><i class="sq" style="background:${PIPES[p].color}"></i>${PIPES[p].name}</span>`).join("")}</div>`;
}

function drawQualityChart(runs) {
  const host = $("#ch-quality");
  const m = state.benchMetric;
  const names = { "ndcg@10": "NDCG@10", "recall@1": "Recall@1", "recall@5": "Recall@5", "mrr@10": "MRR@10" };
  const rowH = 26, padL = 230, padR = 46, W = 640, H = runs.length * rowH + 28;
  const x = (v) => padL + (W - padL - padR) * v;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${names[m]} by configuration">`;
  s += `<g class="grid">${[0, 0.25, 0.5, 0.75, 1].map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 22}"/><text x="${x(t)}" y="${H - 6}" text-anchor="middle">${t}</text>`).join("")}</g>`;
  runs.forEach((r, i) => {
    const v = r.metrics[m] ?? 0, y0 = i * rowH + 6;
    s += `<g tabindex="0" data-tip="<b>${esc(r.label)}</b><br>${names[m]} ${fmt(v)}">
      <rect x="0" y="${y0 - 3}" width="${W}" height="${rowH}" fill="transparent"/>
      <text class="lbl" x="${padL - 10}" y="${y0 + 13}" text-anchor="end">${esc(r.label.length > 34 ? r.label.slice(0, 33) + "..." : r.label)}</text>
      <path d="${hbarPath(x(0), y0 + 2, x(v) - x(0), 16)}" fill="${PIPES[r.pipeline].color}"/>
      <text class="val" x="${x(v) + 6}" y="${y0 + 14}">${fmt(v)}</text></g>`;
  });
  s += `</svg>`;
  host.innerHTML = `<h2>Retrieval quality</h2>
    <p class="sub">Higher is better. Grey bars are the PCA cosine control, which isolates the cost of compressing embeddings to the qubit budget.</p>
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center">${pipelineLegend(runs)}
    ${segmented("benchMetric", Object.entries(names), m)}</div>${s}`;
  host.querySelectorAll("[data-set='benchMetric']").forEach((b) => b.addEventListener("click", () => { state.benchMetric = b.dataset.val; drawQualityChart(runs); }));
  bindTips(host);
}

function drawLatencyChart(runs) {
  const host = $("#ch-latency");
  const rowH = 26, padL = 230, padR = 20, W = 640, H = runs.length * rowH + 30;
  const vals = runs.flatMap((r) => [r.latency_ms.p50, r.latency_ms.p95, r.latency_ms.mean]).filter((v) => v > 0);
  const lo = 10 ** Math.floor(Math.log10(Math.min(...vals))), hi = 10 ** Math.ceil(Math.log10(Math.max(...vals)));
  const x = (v) => padL + (W - padL - padR) * (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Query latency by configuration, log scale">`;
  s += `<g class="grid">${logTicks(lo, hi).map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H - 24}"/><text x="${x(t)}" y="${H - 8}" text-anchor="middle">${t >= 1 ? t.toLocaleString() : t} ms</text>`).join("")}</g>`;
  runs.forEach((r, i) => {
    const y0 = i * rowH + 15, L = r.latency_ms;
    s += `<g tabindex="0" data-tip="<b>${esc(r.label)}</b><br>mean ${fmtMs(L.mean)} ms<br>median ${fmtMs(L.p50)} ms, p95 ${fmtMs(L.p95)} ms">
      <rect x="0" y="${y0 - 12}" width="${W}" height="${rowH}" fill="transparent"/>
      <text class="lbl" x="${padL - 10}" y="${y0 + 4}" text-anchor="end">${esc(r.label.length > 34 ? r.label.slice(0, 33) + "..." : r.label)}</text>
      <line x1="${x(L.p50)}" x2="${x(L.p95)}" y1="${y0}" y2="${y0}" stroke="${PIPES[r.pipeline].color}" stroke-width="2" stroke-linecap="round" opacity=".55"/>
      <circle cx="${x(L.mean)}" cy="${y0}" r="5" fill="${PIPES[r.pipeline].color}" stroke="var(--surface)" stroke-width="2"/></g>`;
  });
  s += `</svg>`;
  host.innerHTML = `<h2>Query latency</h2><p class="sub">Dot is the mean, line spans median to 95th percentile. Log scale; excludes the shared query embedding step.</p>${pipelineLegend(runs)}${s}`;
  bindTips(host);
}

function drawMemoryChart(runs) {
  const host = $("#ch-mem");
  const rowH = 26, padL = 230, padR = 70, W = 640, H = runs.length * rowH + 28;
  const max = niceMax(Math.max(...runs.map((r) => r.memory.index_bytes)));
  const x = (v) => padL + (W - padL - padR) * v / max;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Index memory by configuration">`;
  s += `<g class="grid">${[0, 0.5, 1].map((t) => `<line x1="${x(t * max)}" x2="${x(t * max)}" y1="0" y2="${H - 22}"/><text x="${x(t * max)}" y="${H - 6}" text-anchor="middle">${fmtBytes(t * max)}</text>`).join("")}</g>`;
  runs.forEach((r, i) => {
    const v = r.memory.index_bytes, y0 = i * rowH + 6;
    s += `<g tabindex="0" data-tip="<b>${esc(r.label)}</b><br>index ${fmtBytes(v)}<br>peak per query ${fmtBytes(r.memory.peak_query_bytes)}">
      <rect x="0" y="${y0 - 3}" width="${W}" height="${rowH}" fill="transparent"/>
      <text class="lbl" x="${padL - 10}" y="${y0 + 13}" text-anchor="end">${esc(r.label.length > 34 ? r.label.slice(0, 33) + "..." : r.label)}</text>
      <path d="${hbarPath(x(0), y0 + 2, x(v) - x(0), 16)}" fill="${PIPES[r.pipeline].color}"/>
      <text class="val" x="${x(v) + 6}" y="${y0 + 14}">${fmtBytes(v)}</text></g>`;
  });
  s += `</svg>`;
  host.innerHTML = `<h2>Index memory</h2><p class="sub">Stored vectors or simulated statevectors for the whole corpus. Hybrid keeps both the classical index and the quantum states.</p>${pipelineLegend(runs)}${s}`;
  bindTips(host);
}

function lineChart({ series, xLog = false, yLog = false, xLabel, yLabel, yMin, yMax, xTicks, W = 640, H = 260, markers = [] }) {
  const padL = 56, padR = 70, padT = 10, padB = 40;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1])).filter((v) => !yLog || v > 0);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = yMin ?? (yLog ? 10 ** Math.floor(Math.log10(Math.min(...ys))) : 0);
  const y1 = yMax ?? (yLog ? 10 ** Math.ceil(Math.log10(Math.max(...ys))) : niceMax(Math.max(...ys)));
  const fx = (v) => padL + (W - padL - padR) * (xLog ? (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : (v - x0) / (x1 - x0));
  const fy = (v) => padT + (H - padT - padB) * (1 - (yLog ? (Math.log10(Math.max(v, y0)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) : (v - y0) / (y1 - y0)));
  const yt = yLog ? logTicks(y0, y1) : [0, 0.25, 0.5, 0.75, 1].map((t) => y0 + t * (y1 - y0));
  const xt = (xTicks || (xLog ? logTicks(x0, x1) : [...new Set(xs)])).filter((t) => t >= x0 * 0.999 && t <= x1 * 1.001);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} against ${esc(xLabel)}">`;
  s += `<g class="grid">${yt.map((t) => `<line x1="${padL}" x2="${W - padR}" y1="${fy(t)}" y2="${fy(t)}"/><text x="${padL - 8}" y="${fy(t) + 4}" text-anchor="end">${yLog ? supNum(t) : (+t.toFixed(2)).toString()}</text>`).join("")}</g>`;
  s += xt.map((t) => `<text x="${fx(t)}" y="${H - padB + 18}" text-anchor="middle">${xLog && t >= 10000 ? supNum(t) : t.toLocaleString()}</text>`).join("");
  s += `<text x="${(padL + W - padR) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>`;
  s += `<text x="12" y="${padT + 4}" transform="rotate(-90 12 ${padT + 4})" text-anchor="end">${esc(yLabel)}</text>`;
  markers.forEach((m) => {
    const nearRight = fx(m.x) > (W - padR) - 120;
    s += `<line x1="${fx(m.x)}" x2="${fx(m.x)}" y1="${padT}" y2="${H - padB}" stroke="var(--rule-strong)"/><text x="${fx(m.x) + (nearRight ? -6 : 6)}" y="${padT + 12}" text-anchor="${nearRight ? "end" : "start"}" class="lbl">${esc(m.label)}</text>`;
  });
  series.forEach((se) => {
    const d = se.points.map((p, i) => `${i ? "L" : "M"}${fx(p[0]).toFixed(1)} ${fy(p[1]).toFixed(1)}`).join(" ");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    se.points.forEach((p) => {
      s += `<g tabindex="-1" data-tip="<b>${esc(se.name)}</b><br>${esc(xLabel)} ${p[0].toLocaleString()}<br>${esc(yLabel)} ${yLog ? p[1].toExponential(2) : p[1].toFixed(3)}">
        <circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="11" fill="transparent"/>
        <circle cx="${fx(p[0])}" cy="${fy(p[1])}" r="4" fill="${se.color}" stroke="var(--surface)" stroke-width="2"/></g>`;
    });
    const last = se.points[se.points.length - 1];
    if (series.length > 1 && series.length <= 4) s += `<text class="lbl" x="${fx(last[0]) + 8}" y="${fy(last[1]) + 4}">${esc(se.short || se.name)}</text>`;
  });
  return s + `</svg>`;
}
function legendFor(series) {
  if (series.length < 2) return "";
  return `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>`;
}

function drawVarianceChart(rows) {
  const host = $("#ch-variance");
  const series = [{ name: "Explained variance", color: "var(--s1)", points: rows.map((r) => [r.dim, r.explained_variance]) }];
  host.innerHTML = `<h2>Information kept after PCA</h2>
    <p class="sub">Share of embedding variance retained by the first d principal components. Angle and IQP encodings use d = N qubits; amplitude encoding uses d = 2^N.</p>
    ${lineChart({ series, xLog: true, xLabel: "PCA dimensions", yLabel: "Variance kept", yMin: 0, yMax: 1, xTicks: rows.map((r) => r.dim),
      markers: [{ x: 8, label: "8 qubits, angle" }, { x: 256, label: "8 qubits, amplitude" }].filter((m) => rows.some((r) => r.dim === m.x)) })}`;
  bindTips(host);
}

function drawConcentration(rows) {
  const host = $("#ch-conc");
  const bws = [...new Set(rows.map((r) => r.bandwidth))];
  const bw = bws.includes(state.concBandwidth) ? state.concBandwidth : bws[0];
  const pick = (src, enc) => rows.filter((r) => (r.source || "corpus") === src && r.encoding === enc && r.bandwidth === bw)
    .sort((a, b) => a.n_qubits - b.n_qubits).map((r) => [r.n_qubits, Math.max(r.variance, 1e-12)]);
  const series = [
    { name: "Angle, corpus features", short: "Angle", color: "var(--s1)", points: pick("corpus", "angle") },
    { name: "IQP, corpus features", short: "IQP", color: "var(--s2)", points: pick("corpus", "iqp") },
    { name: "IQP, random features", short: "Random", color: "var(--c-control)", points: pick("random", "iqp") },
  ].filter((x) => x.points.length);
  host.innerHTML = `<h2>Kernel concentration</h2>
    <p class="sub">Variance of fidelities between different inputs. When it collapses towards zero, every document looks equally similar to every query. Random features show the textbook exponential collapse.</p>
    <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center">${legendFor(series)}
      <label class="field" style="flex-direction:row;align-items:center;gap:8px"><span>Bandwidth</span>${segmented("concBw", bws.map((b) => [b, b.toFixed(2)]), bw)}</label></div>
    ${lineChart({ series, yLog: true, xLabel: "Qubits", yLabel: "Fidelity variance" })}`;
  host.querySelectorAll("[data-set='concBw']").forEach((b) => b.addEventListener("click", () => { state.concBandwidth = parseFloat(b.dataset.val); drawConcentration(rows); }));
  bindTips(host);
}

function drawEstimator(est) {
  const host = $("#ch-est");
  const series = [
    { name: "Plain sampling", short: "Sampling", color: "var(--s1)", points: est.sampling.map((r) => [r.oracle_calls, r.mean_abs_error]) },
    { name: "Iterative amplitude estimation", short: "IQAE", color: "var(--s2)", points: est.iqae.map((r) => [Math.round(r.oracle_calls), r.mean_abs_error]) },
  ];
  host.innerHTML = `<h2>Estimator efficiency</h2>
    <p class="sub">Mean absolute error of the fidelity estimate against total circuit calls, over ${est.trials} random fidelities. Sampling improves as 1/&radic;calls; IQAE as 1/calls.</p>
    ${legendFor(series)}
    ${lineChart({ series, xLog: true, yLog: true, xLabel: "Circuit calls", yLabel: "Mean abs. error" })}`;
  bindTips(host);
}

// ================================================================= CORPUS
async function renderCorpus() {
  const custom = state.status?.corpus === "custom";
  view.innerHTML = `
    <header class="page-head">
      <h1>Corpus</h1>
      <p>The documents every pipeline searches. Upload your own files to ask questions about them; text is extracted,
      split into overlapping passages and all three pipelines are rebuilt on it.</p>
    </header>
    <div class="corpus-layout">
      <section class="panel">
        <form class="corpus-tools" id="cfilter">
          <label class="visually-hidden" for="cf">Filter documents</label>
          <input id="cf" placeholder="Filter by title or text" value="${esc(state.corpus.q)}">
          <button class="btn secondary small" type="submit">Filter</button>
        </form>
        <div id="clist"><p class="loading" style="padding:16px"><span class="spinner"></span>Loading documents.</p></div>
      </section>
      <div style="display:grid;gap:16px">
        <section class="panel">
          <form class="add-doc" id="upform">
            <h2>Upload files</h2>
            <label class="dropzone" id="drop">
              <input type="file" id="files" multiple accept=".pdf,.txt,.md,.docx" class="visually-hidden">
              <span class="dz-title">Choose files or drop them here</span>
              <span class="hint">PDF, Word (.docx), text (.txt) or Markdown (.md), up to 25 MB each</span>
            </label>
            <ul class="filelist" id="filelist"></ul>
            <fieldset class="modes">
              <legend class="visually-hidden">What to search after uploading</legend>
              <label><input type="radio" name="mode" value="replace" checked> <span><b>Search only these files</b><br><span class="hint">Best for asking questions about your document. Benchmark questions are set aside until you restore the corpus.</span></span></label>
              <label><input type="radio" name="mode" value="add"> <span><b>Add to the current corpus</b><br><span class="hint">Your passages compete with the ${custom ? "documents already loaded" : "600 SQuAD paragraphs"}.</span></span></label>
            </fieldset>
            <div style="display:flex;gap:10px;flex-wrap:wrap">
              <button class="btn" type="submit" id="upbtn" disabled>Upload and index</button>
              <button class="btn secondary" type="button" id="reset">Restore benchmark corpus</button>
            </div>
            <div id="upmsg" aria-live="polite"></div>
          </form>
        </section>
        <section class="panel">
          <details class="more paste">
            <summary>Paste text instead</summary>
            <form class="add-doc" id="addform" style="padding:8px 0 0">
              <label class="field"><span>Title</span><input name="title" maxlength="300" placeholder="Lecture notes, week 3"></label>
              <label class="field"><span>Text</span><textarea name="text" required minlength="20" placeholder="Paste the text to index"></textarea></label>
              <div><button class="btn" type="submit">Add to corpus</button></div>
              <p class="hint" id="addmsg"></p>
            </form>
          </details>
        </section>
        <section class="panel topics" id="topics"></section>
      </div>
    </div>`;

  let picked = [];
  const listFiles = () => {
    $("#filelist").innerHTML = picked.map((f) => `<li><span>${esc(f.name)}</span><span class="muted num">${fmtBytes(f.size)}</span></li>`).join("");
    $("#upbtn").disabled = !picked.length;
  };
  const accept = (fileList) => {
    const ok = [...fileList].filter((f) => /\.(pdf|txt|md|docx)$/i.test(f.name));
    const skipped = fileList.length - ok.length;
    picked = ok;
    listFiles();
    $("#upmsg").innerHTML = skipped ? `<p class="hint">${skipped} file${skipped === 1 ? " was" : "s were"} skipped: only PDF, DOCX, TXT and MD are supported.</p>` : "";
  };
  $("#files").addEventListener("change", (e) => accept(e.target.files));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => accept(e.dataTransfer.files));

  const toBase64 = (file) => new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] || "");
    r.onerror = () => reject(new Error(`Could not read ${file.name}`));
    r.readAsDataURL(file);
  });

  $("#upform").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!picked.length) return;
    const btn = $("#upbtn"), msg = $("#upmsg");
    const mode = new FormData(e.target).get("mode");
    btn.disabled = true;
    msg.innerHTML = `<p class="loading"><span class="spinner"></span>Extracting text and rebuilding the indexes. Large PDFs can take a minute.</p>`;
    try {
      const files = await Promise.all(picked.map(async (f) => ({ name: f.name, data_base64: await toBase64(f) })));
      const r = await api("/api/corpus/upload", { files, mode });
      msg.innerHTML = `
        <p class="hint" style="color:var(--good)">Indexed ${r.added_chunks.toLocaleString()} passage${r.added_chunks === 1 ? "" : "s"}.
        ${r.corpus === "custom" ? "Searches now cover only your uploaded files." : `The corpus now has ${r.documents.toLocaleString()} documents.`}
        <a href="#/retrieve">Ask a question</a></p>
        <ul class="filelist">${r.files.map((f) => `<li><span>${esc(f.name)}</span><span class="muted num">${f.sections} section${f.sections === 1 ? "" : "s"}, ${f.words.toLocaleString()} words</span></li>`).join("")}</ul>
        ${r.errors.length ? `<div class="error">${r.errors.map(esc).join("<br>")}</div>` : ""}`;
      picked = []; listFiles(); $("#files").value = "";
      afterCorpusChange();
    } catch (err) {
      msg.innerHTML = `<div class="error">${esc(err.message)}</div>`;
      btn.disabled = false;
    }
  });

  $("#cfilter").addEventListener("submit", (e) => { e.preventDefault(); state.corpus = { offset: 0, q: $("#cf").value.trim() }; loadCorpus(); });
  $("#addform").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target, msg = $("#addmsg");
    const btn = f.querySelector("button[type=submit]");
    btn.disabled = true; msg.textContent = "Adding and rebuilding the indexes.";
    try {
      const r = await api("/api/corpus", { documents: [{ title: f.title.value, text: f.text.value }] });
      msg.textContent = `Added ${r.added_chunks} passage${r.added_chunks === 1 ? "" : "s"}. The corpus now has ${r.documents.toLocaleString()} documents.`;
      f.reset(); afterCorpusChange();
    } catch (err) { msg.textContent = err.message; }
    btn.disabled = false;
  });
  $("#reset").addEventListener("click", async () => {
    $("#upmsg").innerHTML = `<p class="loading"><span class="spinner"></span>Restoring the benchmark corpus.</p>`;
    try {
      const r = await api("/api/corpus/reset", {});
      $("#upmsg").innerHTML = `<p class="hint">Restored ${r.documents.toLocaleString()} benchmark documents and their questions.</p>`;
      afterCorpusChange();
    } catch (err) { $("#upmsg").innerHTML = `<div class="error">${esc(err.message)}</div>`; }
  });
  loadCorpus();
}

function afterCorpusChange() {
  state.lastSearch = null;
  state.samples = [];
  state.corpus = { offset: 0, q: "" };
  loadStatus();
  loadCorpus();
}

async function loadCorpus() {
  const { offset, q } = state.corpus;
  const limit = 20;
  try {
    const d = await api(`/api/corpus?offset=${offset}&limit=${limit}&q=${encodeURIComponent(q)}`);
    $("#clist").innerHTML = d.items.length ? `
      <ul class="doclist">${d.items.map((it) => `<li>
        <h3>${esc(it.title)}</h3>
        <p>${esc(it.text)}</p>
        <div class="meta">Document ${it.index + 1}, ${it.words} words, id <span class="mono">${esc(it.id)}</span></div></li>`).join("")}</ul>
      <div class="pager"><span>${offset + 1} to ${Math.min(offset + limit, d.total)} of ${d.total.toLocaleString()}</span>
        <span style="display:flex;gap:8px"><button class="btn secondary small" id="prev" ${offset === 0 ? "disabled" : ""}>Previous</button>
        <button class="btn secondary small" id="next" ${offset + limit >= d.total ? "disabled" : ""}>Next</button></span></div>`
      : `<p class="empty">No documents match "${esc(q)}". Clear the filter to see the whole corpus.</p>`;
    $("#prev")?.addEventListener("click", () => { state.corpus.offset = Math.max(0, offset - limit); loadCorpus(); });
    $("#next")?.addEventListener("click", () => { state.corpus.offset = offset + limit; loadCorpus(); });
    $("#topics").innerHTML = `<h2>Topics</h2><p class="hint">${d.topics.length} ${state.status?.corpus === "custom" ? "uploaded file" : "source article"}${d.topics.length === 1 ? "" : "s"}</p>
      <ul>${d.topics.map((t) => `<li><span>${esc(t.topic)}</span><span class="num muted">${t.count}</span></li>`).join("")}</ul>`;
  } catch (err) { $("#clist").innerHTML = `<div class="error">${esc(err.message)}</div>`; }
}

// ================================================================= METHOD
function renderMethod() {
  const s = state.status;
  view.innerHTML = `
    <header class="page-head">
      <h1>Method</h1>
      <p>QuantumRAG asks whether the retrieval stage of retrieval-augmented generation can use quantum state
      similarity instead of cosine similarity, and measures what that costs on today's simulators.</p>
    </header>
    <div class="pipes">
      <section class="panel col" data-p="classical"><h3>Classical baseline</h3><ol>
        <li>Embed the query with ${s?.embedder_backend === "minilm" ? "all-MiniLM-L6-v2" : "the active embedder"} (384-d).</li>
        <li>Exact inner product over normalised vectors (${s?.faiss ? "FAISS IndexFlatIP" : "NumPy, FAISS when installed"}).</li>
        <li>Return the top K by cosine similarity.</li></ol></section>
      <section class="panel col" data-p="quantum"><h3>Pure quantum</h3><ol>
        <li>Compress the embedding with PCA to N features (2^N for amplitude encoding).</li>
        <li>Prepare |&psi;(x)&rang; with the angle, IQP or amplitude feature map.</li>
        <li>Estimate fidelity with every document by kernel circuit, SWAP test or IQAE.</li>
        <li>Return the top K by fidelity.</li></ol></section>
      <section class="panel col" data-p="hybrid"><h3>Hybrid two-stage</h3><ol>
        <li>Shortlist K<sub>c</sub> candidates with classical cosine search.</li>
        <li>Re-rank only those candidates by quantum fidelity.</li>
        <li>Return the top K<sub>q</sub>. Quantum cost grows with K<sub>c</sub>, not corpus size.</li></ol></section>
    </div>
    <div class="prose">
      <h2>Fidelity estimators</h2>
      <p>All three estimators measure the same quantity, K(x, y) = |&lang;&psi;(x)|&psi;(y)&rang;|&sup2;.
      The <b>kernel circuit</b> runs U(x) followed by U(y)<sup>&dagger;</sup> and reads the probability of all zeros.
      The <b>SWAP test</b> needs 2N + 1 qubits but no inverse circuit: the ancilla reads 0 with probability (1 + K) / 2.
      <b>Iterative amplitude estimation</b> (Grinko et al., 2021) amplifies that probability with Grover powers and
      reaches error &epsilon; with roughly 1/&epsilon; circuit calls instead of 1/&epsilon;&sup2; shots.</p>
      <h2>How the simulator is used</h2>
      <p>Circuits run on PennyLane's <code>default.qubit</code> simulator. Document states are simulated once by the
      feature-map circuit and cached; each query is then compared against the cached states using the measurement
      statistics of the chosen estimator, with finite shots drawn from that distribution when a shot count is set.
      The unit tests check that this matches executing the full kernel, SWAP-test and Grover circuits pair by pair.
      Full-circuit execution is also available in the settings for the hybrid pipeline.</p>
      <h2>Reading the results honestly</h2>
      <p>Latency figures are simulator-relative: they measure a classical CPU emulating quantum circuits, so they say
      nothing about speed on quantum hardware. The PCA cosine control isolates how much accuracy is lost to compressing
      embeddings into a few qubits, which is separate from any effect of the quantum kernel itself.</p>
      <h2>Environment</h2>
      <p>${s ? `${esc(s.dataset)}, ${s.documents.toLocaleString()} documents. Embeddings: ${esc(s.embedder)}.
        Index: ${esc(s.index)}. PennyLane ${esc(s.pennylane)}. Qiskit ${s.qiskit ? esc(s.qiskit) + ", used to cross-check the circuits" : "is not installed; install it to cross-check the circuits on a second framework"}.
        Answer generation: ${esc(s.llm.model)}.` : "Status unavailable."}</p>
      ${s?.qiskit ? `<p><button class="btn secondary small" id="xcheck" type="button">Cross-check circuits with Qiskit</button></p><div id="xout"></div>` : ""}
    </div>`;
  $("#xcheck")?.addEventListener("click", async () => {
    $("#xout").innerHTML = `<p class="loading"><span class="spinner"></span>Running the same circuits in Qiskit.</p>`;
    try {
      const r = await api("/api/crosscheck?encoding=angle&n_qubits=4");
      $("#xout").innerHTML = r.available ? `<p>Largest difference between PennyLane and Qiskit fidelities over ${r.rows.length} document pairs: <b>${r.max_abs_deviation.toExponential(2)}</b>.</p>` : `<p>${esc(r.detail)}</p>`;
    } catch (err) { $("#xout").innerHTML = `<div class="error">${esc(err.message)}</div>`; }
  });
}

// ------------------------------------------------------------------- boot
route();
loadStatus().then(() => { if ((location.hash || "#/retrieve").startsWith("#/method")) renderMethod(); });
