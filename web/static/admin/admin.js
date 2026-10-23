/* ScholarGraph Control Deck — rebuilt client (no deps, same API contract) */
'use strict';
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const short = (s, n = 16) => { const t = String(s ?? ''); return t.length <= n ? t : t.slice(0, n) + '…'; };
const num = (v, d = 0) => { const n = Number(v); return Number.isFinite(n) ? n : d; };
const money = (v) => '$' + num(v, 0).toFixed(2);
const fmtInt = (v) => Math.round(num(v, 0)).toLocaleString('en-US');

function _copyMap(target, source, skipMasked = false) {
  Object.entries(source || {}).forEach(([key, value]) => {
    if (value === undefined) return;
    if (skipMasked && typeof value === 'string' && (value.includes('…') || value.startsWith('••'))) return;
    target[String(key).toUpperCase()] = value;
  });
}

function apiErrorMessage(data, fallback) {
  if (!data || typeof data !== 'object') return fallback;
  const d = data.detail;
  if (typeof d === 'string' && d) return d;
  if (Array.isArray(d) && d.length) return d[0]?.msg || d[0]?.message || fallback;
  if (d && typeof d === 'object') return d.error || d.message || d.msg || fallback;
  if (typeof data.error === 'string' && data.error) return data.error;
  if (typeof data.message === 'string' && data.message) return data.message;
  return fallback;
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(apiErrorMessage(data, r.statusText || ('HTTP ' + r.status)));
  return data;
}

/* ---------- state ---------- */
let dashboard = {}, runStatus = {}, scratch = {}, selectedRunId = null;
let lastPhase = null, pollTimer = null, pollMs = 10000, backoff = 0;
let activeView = 'overview', activeSettingsGroup = 'LLM', settingsCache = {}, keysCache = {};
let spendHistory = [];
const phases = ['intake', 'topic_discovery', 'hypothesis_debate', 'planning', 'data_validation', 'writing_narrative', 'engineering', 'independent_validation', 'writing_results', 'supervision', 'editing'];
const TITLES = { overview: 'Run overview', experiments: 'Experiments', evidence: 'Evidence and lineage', agents: 'Agent registry', paper: 'Manuscript workspace', activity: 'Activity', runs: 'Run history', settings: 'System settings' };
const FALLBACK_RECIPE = phases;

// A deliberate design-mode fixture: use `/?demo=1` to review the interface
// without a provider, run ledger, or network. Production data always wins.
function demoDashboard() {
  return { run_id: 'aurora-24', phase: 'hypothesis_debate', readiness: 'L1',
    release: { readiness: 'L1', status: 'in review', reason: 'Evidence graph is assembling' },
    tracker_stats: { cost_estimate_usd: 8.42, llm_calls: 47, llm_tokens_in: 183420, llm_tokens_out: 14281, started_at: new Date(Date.now() - 17 * 60000).toISOString() },
    recent_messages: [{ type: 'evidence_map', message: 'Cross-paper signals clustered into 3 candidate gaps.' }, { type: 'debate_round', message: 'Challenger requested a stronger falsifiability criterion.' }],
    workspace: { current_phase: 'hypothesis_debate', iteration: 1, phase_steps: { intake: 1, topic_discovery: 4, hypothesis_debate: 2 },
      topics: [
        { title: 'Counterfactual traces for autonomous repository repair', hunt_mode: 'gap', status: 'candidate' },
        { title: 'When agent memory makes evidence less reproducible', hunt_mode: 'try_and_report', status: 'candidate' },
        { title: 'Measuring semantic drift across multi-agent review loops', hunt_mode: 'gap', status: 'candidate' }
      ],
      selected_topic: { title: 'Counterfactual traces for autonomous repository repair' },
      debate_results: [{ topic: 'Counterfactual traces for autonomous repository repair', score: 8.7 }],
      verification_findings: [{ check: 'source coverage', message: 'Four primary sources resolved.', blocking: false }],
      pipeline_recipe: FALLBACK_RECIPE,
      plan: { contributions: ['A reproducible trace-level repair benchmark'] },
      engineer_outputs: { 'trace-probe': { success: true, results: { metrics: { precision: 0.81, recall: 0.74 } } } },
      draft_sections: { Abstract: 'A live demonstration fixture for the ScholarGraph interface.' }
    }
  };
}

/* ---------- toast + confirm ---------- */
let toastTimer = null;
function toast(msg, kind) {
  const el = $('toast'); el.textContent = msg;
  el.classList.remove('ok', 'bad');
  if (kind === 'ok' || kind === 'bad') el.classList.add(kind);
  el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show', 'ok', 'bad'), 2400);
}
function confirmAction(title, message) {
  return new Promise((resolve) => {
    $('confirmTitle').textContent = title; $('confirmMessage').textContent = message;
    $('confirmOverlay').classList.add('show');
    const done = (v) => { $('confirmOverlay').classList.remove('show'); $('confirmOk').onclick = null; $('confirmCancel').onclick = null; resolve(v); };
    $('confirmOk').onclick = () => done(true); $('confirmCancel').onclick = () => done(false);
  });
}

/* ---------- theme ---------- */
function applyTheme(t) { document.documentElement.dataset.theme = t; try { localStorage.setItem('sg-theme', t); } catch {} }
function initTheme() {
  let t = null; try { t = localStorage.getItem('sg-theme'); } catch {}
  if (!t) t = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
  applyTheme(t);
}

/* ---------- nav ---------- */
function showView(name) {
  activeView = name;
  document.querySelectorAll('.nav button').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === name));
  $('viewTitle').textContent = TITLES[name] || name;
  const nextView = $(name); if (nextView) delete nextView.dataset.revealed;
  if (name === 'runs') loadRuns();
  if (name === 'settings') renderSettings();
  if (name === 'agents') renderAgents();
}

/* ---------- polling (slim-first, U1 single transport) ---------- */
function schedulePoll(ms) { clearTimeout(pollTimer); pollTimer = setTimeout(tick, ms); }
async function tick() {
  const q = selectedRunId ? `?run_id=${encodeURIComponent(selectedRunId)}` : '';
  try {
    const slim = await api(`/api/dashboard/slim${q}`);
    const st = await api('/api/run/status').catch(() => ({}));
    runStatus = st || {};
    const running = !!(st && st.running);
    pollMs = running ? 3000 : 10000;
    backoff = 0;
    if (slim.phase !== lastPhase || !dashboard.run_id) {
      lastPhase = slim.phase;
      await refreshFull();
    } else {
      paintStatus(slim, st);
    }
  } catch (e) { backoff = Math.min(backoff + 1, 5); }
  schedulePoll(pollMs + backoff * 2000);
}
async function refreshFull() {
  const q = selectedRunId ? `?run_id=${encodeURIComponent(selectedRunId)}` : '';
  if (new URLSearchParams(location.search).has('demo')) {
    dashboard = demoDashboard(); runStatus = { running: true }; scratch = { lines: ['Demo mode — no API calls are made.'] }; renderAll(); return;
  }
  try {
    const [d, st, sp] = await Promise.all([
      api(`/api/dashboard${q}`), api('/api/run/status').catch(() => ({})),
      api(`/api/scratchpad?limit=30${selectedRunId ? `&run_id=${encodeURIComponent(selectedRunId)}` : ''}`).catch(() => ({})),
    ]);
    dashboard = d || {}; runStatus = st || {}; scratch = sp || {};
    let health = null; try { health = await api('/api/health'); } catch {}
    if (health && $('dbPath')) $('dbPath').textContent = `${health.provider || '?'} · ${health.model || '?'}`;
    renderAll();
  } catch (e) { console.error(e); }
}

/* ---------- status pill ---------- */
function paintStatus(slim, st) {
  const running = !!(st && st.running);
  const pill = $('runStatus'), pulse = $('pulse');
  const label = running ? ('Running · ' + (slim.phase || '…')) : (slim.readiness === 'L3' ? 'Ready' : (slim.phase && slim.phase !== 'idle' ? String(slim.phase).replaceAll('_', ' ') : 'Idle'));
  pill.textContent = label;
  pill.className = 'status-pill' + (running ? ' running' : (slim.readiness === 'L3' ? ' ready' : ''));
  pulse.className = 'pulse' + (running ? ' live' : '');
  const rid = slim.run_id || (dashboard || {}).run_id || '';
  const runBtn = $('runId'); runBtn.textContent = rid ? short(rid, 18) : '—'; runBtn.title = rid || 'no run';
  $('copyRunId').style.display = rid ? '' : 'none';
  $('crumbLine').textContent = rid ? `run ${rid} · ${slim.phase || 'idle'} · readiness ${slim.readiness || '?'}` : 'No active research run.';
}

/* ---------- render root ---------- */
function ws() { return (dashboard && dashboard.workspace && typeof dashboard.workspace === 'object') ? dashboard.workspace : {}; }
function renderAll() {
  const d = dashboard, w = ws();
  const slimPhase = (d.pipeline && d.pipeline.find((p) => p.state === 'active')?.phase) || d.phase || 'idle';
  paintStatus({ run_id: d.run_id, phase: d.phase || slimPhase, readiness: d.readiness || (d.release || {}).readiness }, runStatus);
  $('topicLine').textContent = topicLineText(w);
  renderKpis(d, w);
  renderStepper(w);
  renderFunnel(w);
  renderSpend(d, w);
  renderVerify(d, w);
  renderGate(d, w);
  renderFinal(d, w);
  renderRouting(w);
  renderTopics(w);
  renderActivity(d);
  renderPreflight(w);
  renderSnapshot(w);
  renderExperiments(w);
  renderEvidence(d, w);
  renderPaper(w);
  renderRunsBadge();
  if (activeView === 'runs') loadRuns();
  if (activeView === 'agents') renderAgents();
  if (activeView === 'settings') renderSettings();
}
function topicLineText(w) {
  const t = w.selected_topic;
  if (t && t.title) return `${t.title} · ${w.paper_type || 'auto'} · ${w.hunt_mode || 'try_and_report'}`;
  const n = (w.topics || []).length;
  if (n) return `${n} candidate topic(s) under debate · ${w.paper_type || 'auto'}`;
  const goal = (w.intake_report || {}).goal || w.user_query;
  if (goal) return `Goal: ${String(goal).slice(0, 140)}`;
  return 'No active research run.';
}

/* ---------- KPIs ---------- */
function renderKpis(d, w) {
  const rel = d.release || {};
  $('readiness').textContent = rel.readiness || d.readiness || 'Pending';
  $('readiness').className = 'metric-value' + (String(rel.readiness).startsWith('L3') ? ' good' : (rel.status === 'blocked' ? ' bad' : ''));
  $('readinessNote').textContent = rel.reason || rel.status || 'Waiting for evidence';
  const phase = d.phase || 'Idle';
  $('phaseMetric').textContent = String(phase).replaceAll('_', ' ');
  $('phaseNote').textContent = `run ${short(d.run_id || '—')} · iteration ${w.iteration ?? 0}`;
  const findings = w.verification_findings || [];
  const blocking = findings.filter((f) => f && f.blocking).length;
  $('findingMetric').textContent = String(findings.length);
  $('findingMetric').className = 'metric-value' + (blocking ? ' bad' : (findings.length ? ' warn' : ' good'));
  const stats = d.tracker_stats || d.stats || {};
  const usd = num(stats.cost_estimate_usd ?? stats.usd ?? 0);
  $('spendMetric').textContent = money(usd);
  $('spendNote').textContent = `${fmtInt(stats.llm_calls ?? 0)} calls · ${fmtInt((stats.llm_tokens_in ?? 0) + (stats.llm_tokens_out ?? 0))} tokens`;
  $('costTokens').textContent = fmtInt((stats.llm_tokens_in ?? 0) + (stats.llm_tokens_out ?? 0));
  $('costCalls').textContent = fmtInt(stats.llm_calls ?? 0);
  const started = stats.started_at ? Date.parse(stats.started_at) : NaN;
  $('costDuration').textContent = Number.isFinite(started) ? Math.max(0, Math.round((Date.now() - started) / 60000)) + 'm' : '—';
  const steps = w.phase_steps || {};
  $('costPhases').textContent = String(Object.values(steps).reduce((a, b) => a + num(b, 0), 0) || 0);
}

/* ---------- Research mesh: live systems map, not a flat workflow strip. ---------- */
function recipeOf(w) {
  const r = w.pipeline_recipe;
  if (Array.isArray(r) && r.length) return r;
  return FALLBACK_RECIPE;
}
function renderStepper(w) {
  const recipe = recipeOf(w);
  const cur = w.current_phase || dashboard.phase || '';
  const activeByPhase = { intake: 'intake', topic_discovery: 'signals', hypothesis_debate: 'debate', planning: 'plan', data_validation: 'data', writing_narrative: 'manuscript', engineering: 'arena', independent_validation: 'evidence', writing_results: 'manuscript', supervision: 'supervisor', meta_evaluation: 'supervisor', editing: 'manuscript' };
  const active = cur && cur !== 'idle' ? activeByPhase[cur] : null;
  const nodes = [['intake', 'Research brief', 'Intent · budgets · scope', '✦'], ['policy', 'Policy + config', 'Keys · capability · budgets', '⊹'], ['gateway', 'Source + model gateway', 'LLM · arXiv · OpenAlex', '◌'], ['signals', 'Signal mining', 'Gaps · novelty · structure', '⌘'], ['debate', 'Debate chamber', 'Proposer ↔ challenger', '↔'], ['plan', 'Experiment design', 'Claims · controls · tests', '⌁'], ['data', 'Data + instrument', 'Packs · contracts · validity', '▦'], ['arena', 'Engineering arena', 'Write → run → repair', '⟐'], ['evidence', 'Evidence firewall', 'Replay · verify · cite', '✓'], ['manuscript', 'Manuscript studio', 'Narrative · results · edit', '¶'], ['supervisor', 'Supervisor + editor', 'Referee · repairs · release', '◈'], ['ledger', 'Memory + artifact ledger', 'Runs · lessons · provenance', '◎']];
  const doneIdx = recipe.indexOf(cur);
  const phaseForNode = { intake: 0, policy: 0, gateway: 1, signals: 1, debate: 2, plan: 3, data: 4, arena: 6, evidence: 7, manuscript: 8, supervisor: 9, ledger: 10 };
  const nodeMarkup = nodes.map(([id, name, sub, icon]) => {
    const state = id === active ? (w.terminal_error ? 'blocked' : 'active') : (doneIdx > (phaseForNode[id] ?? 99) ? 'complete' : 'waiting');
    return `<button class="mesh-node ${state}" data-mesh-node="${id}" type="button" aria-label="Inspect ${esc(name)}"><span class="mesh-icon">${icon}</span><span class="mesh-node-copy"><b>${esc(name)}</b><small>${esc(sub)}</small></span><span class="mesh-state">${state === 'active' ? 'working' : state === 'complete' ? 'passed' : state === 'blocked' ? 'blocked' : 'queued'}</span></button>`;
  }).join('');
  const messages = (dashboard.recent_messages || dashboard.events_tail || []).slice(-3).reverse().map((m) => typeof m === 'string' ? m : (m.message_human || m.message || m.type || '')).filter(Boolean);
  const transfers = messages.length ? messages : (active ? [`${cur.replaceAll('_', ' ')} is receiving a verified handoff`, `${active === 'debate' ? 'Evidence map' : 'Research brief'} → ${nodes.find(n => n[0] === active)?.[1] || 'research mesh'}`] : ['Mesh is quiet — add a research brief to begin a new run.']);
  $('recipeStrip').innerHTML = `<div class="mesh-orbit" aria-hidden="true"></div><svg class="mesh-wires" viewBox="0 0 1000 260" preserveAspectRatio="none" aria-hidden="true"><path d="M115 55 C205 22 270 22 360 55 S505 90 500 130 S445 195 500 210 S710 210 810 170"/><path d="M360 55 C470 25 560 30 640 65 S735 118 810 170"/><path d="M115 55 C170 145 275 190 500 210"/><path d="M500 130 C610 120 715 95 810 170"/></svg><div class="mesh-nodes">${nodeMarkup}</div><div class="mesh-stream"><span class="stream-kicker"><i></i> live handoffs</span>${transfers.map((t, i) => `<button class="stream-item" type="button" data-mesh-node="${i === 0 ? active : 'sources'}"><span>${i === 0 ? 'NOW' : 'FLOW'}</span>${esc(short(t, 104))}<em>→</em></button>`).join('')}</div>`;
  const routed = w.routing_provenance || {};
  const defaultMessage = active ? `${nodes.find(n => n[0] === active)?.[1] || 'Research mesh'} is ${w.terminal_error ? 'blocked' : 'actively receiving'} ${active === 'gateway' ? 'literature and model signals' : 'a routed handoff'} · type=${w.paper_type || 'auto'} · mode=${w.hunt_mode || 'try_and_report'}` : 'System standing by. Add a research brief to route intent through the mesh.';
  $('recipePreview').textContent = defaultMessage + (routed.why ? ` · ${routed.why}` : '') + (w.terminal_error ? ` · ${String(w.terminal_error).slice(0, 120)}` : '');
  $('recipeTag').textContent = active ? `${cur.replaceAll('_', ' ')} live` : 'Standing by';
  $('recipeStrip').querySelectorAll('[data-mesh-node]').forEach((el) => el.onclick = () => {
    const node = nodes.find(n => n[0] === el.dataset.meshNode); if (!node) return;
    $('recipeStrip').querySelectorAll('.mesh-node').forEach((n) => n.classList.toggle('inspected', n.dataset.meshNode === node[0]));
    $('recipePreview').textContent = `${node[1]} · ${node[2]}. ${node[0] === active ? 'This component is active now.' : 'Inspecting its latest handoff and state.'}`;
  });
}

/* ---------- funnel ---------- */
function renderFunnel(w) {
  const topics = (w.topics || []).length;
  const debated = (w.debate_results || []).length;
  const planned = w.plan ? 1 : 0;
  const eng = Object.keys(w.engineer_outputs || {}).length;
  const sections = Object.keys(w.draft_sections || {}).length;
  const rows = [['Topics', topics, topics], ['Debated', debated, Math.max(topics, 1)], ['Planned', planned, 1], ['Experiments out', Math.min(eng, 8), 8], ['Sections', Math.min(sections, 10), 10]];
  const maxV = Math.max(1, topics, debated);
  $('funnelRows').innerHTML = rows.map(([l, v, cap]) => {
    const pct = Math.max(4, Math.round((num(v) / Math.max(maxV, 1)) * 100));
    return `<div class="row"><div><strong>${esc(l)}</strong><div class="bar" style="margin-top:6px;min-width:140px"><i style="width:${pct}%"></i></div></div><span class="pill">${fmtInt(v)}</span></div>`;
  }).join('');
}

/* ---------- spend ---------- */
function renderSpend(d, w) {
  const stats = d.tracker_stats || d.stats || {};
  const usd = num(stats.cost_estimate_usd ?? 0), calls = num(stats.llm_calls ?? 0), toks = (num(stats.llm_tokens_in ?? 0) + num(stats.llm_tokens_out ?? 0));
  const capCalls = 2000, capUsd = 500;
  $('spendTag').textContent = usd >= capUsd ? 'over budget' : usd >= capUsd * 0.8 ? 'near cap' : 'within budget';
  $('spendTag').className = 'tag ' + (usd >= capUsd ? 'bad' : usd >= capUsd * 0.8 ? 'warn' : 'good');
  $('spendRows').innerHTML =
    kv('Spend', `${money(usd)} / $${capUsd}`) +
    kv('LLM calls', `${fmtInt(calls)} / ${fmtInt(capCalls)}`) +
    kv('Tokens', fmtInt(toks));
  const steps = w.phase_steps || {};
  $('stepCounter').textContent = `phase steps: ${Object.entries(steps).map(([k, v]) => `${k}×${v}`).join(' · ') || '—'}`;
  spendHistory.push(usd); if (spendHistory.length > 40) spendHistory.shift();
  drawSpark();
}
function kv(k, v) { return `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span><span></span></div>`; }
function drawSpark() {
  const c = $('spendChart'); if (!c) return;
  const ctx = c.getContext('2d'); const W = c.width = c.clientWidth * 2 || 600, H = c.height = 144;
  ctx.clearRect(0, 0, W, H);
  if (spendHistory.length < 2) { ctx.fillStyle = '#888'; ctx.font = '20px sans-serif'; ctx.fillText('spend trend appears during runs', 16, H / 2); return; }
  const max = Math.max(...spendHistory, 1);
  ctx.beginPath();
  spendHistory.forEach((v, i) => { const x = (i / (spendHistory.length - 1)) * (W - 16) + 8, y = H - 12 - (v / max) * (H - 28); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = '#176b4b'; ctx.lineWidth = 3; ctx.stroke();
}

/* ---------- verification + gate + final ---------- */
function renderVerify(d, w) {
  const v = w.results_verification || {}, repro = w.reproducibility || {};
  const rows = [];
  const push = (ok, label, detail) => rows.push(`<div class="row"><div><strong>${esc(label)}</strong><small>${esc(detail || '')}</small></div><span class="pill ${ok ? 'yes' : 'no'}">${ok ? 'pass' : 'open'}</span></div>`);
  push(repro.passed, 'Reproducibility', (repro.note || repro.message || '').slice(0, 120));
  const keys = Object.keys(v);
  if (!keys.length) rows.push('<div class="empty">No verification results yet — they appear after independent validation.</div>');
  keys.slice(0, 6).forEach((k) => { const r = v[k] || {}; push(r.passed, k, (r.note || r.message || '').slice(0, 120)); });
  $('verifyRows').innerHTML = rows.join('');
  $('verifyTag').textContent = keys.length ? 'checked' : 'pending';
  $('verifyTag').className = 'tag ' + (keys.length ? 'good' : '');
}
function renderGate(d, w) {
  const rel = d.release || {};
  const noRun = !d.run_id && (!d.phase || d.phase === 'idle');
  const tag = $('gateTag');
  tag.textContent = noRun ? 'Not started' : (rel.status || 'Incomplete');
  tag.className = 'tag ' + (!noRun && rel.status === 'ready' ? 'good' : (!noRun && rel.status === 'blocked' ? 'bad' : (!noRun && rel.needs_approval ? 'warn' : '')));
  const missing = rel.missing || [];
  const idleHtml = '<div class="empty">No run yet — start one to open the release gate.</div>';
  $('gateRows').innerHTML = noRun ? idleHtml :
    kv('Readiness', rel.readiness || 'L1') +
    kv('Status', rel.status || 'incomplete') +
    (rel.reason ? `<div class="row"><div><strong>Reason</strong><small>${esc(rel.reason).slice(0, 220)}</small></div></div>` : '') +
    (missing.length ? `<div class="row"><div><strong>Missing</strong><small>${esc(missing.join(', ')).slice(0, 220)}</small></div></div>` : (rel.needs_approval ? '<div class="empty">Evidence complete — waiting on your approval below.</div>' : '<div class="empty">No missing items reported.</div>'));
  const findings = w.verification_findings || [];
  $('findingTag').textContent = String(findings.length);
  // Approve lives here AND in Final review — visible only when it can succeed.
  const gb = $('approveBtnGate');
  if (rel.needs_approval && !noRun) { gb.style.display = ''; gb.disabled = false; }
  else { gb.style.display = 'none'; }
  $('gateHint').textContent = rel.needs_approval && !noRun ? 'Gate is green — your approval releases it.' : '';
}
function renderFinal(d, w) {
  const rel = d.release || {};
  const noRun = !d.run_id && (!d.phase || d.phase === 'idle');
  $('finalTag').textContent = noRun ? 'not started' : (rel.status || 'not ready');
  $('finalTag').className = 'tag ' + (!noRun && rel.status === 'ready' ? 'good' : (!noRun && rel.status === 'blocked' ? 'bad' : (!noRun && rel.needs_approval ? 'warn' : '')));
  const fp = w.final_paper || {};
  $('finalLeft').innerHTML = kv('Paper type', w.paper_type || '—') + kv('Sections', Object.keys(w.draft_sections || {}).length) + kv('Human approved', w.human_approved ? 'yes' : 'no');
  $('finalRight').innerHTML = kv('LaTeX', w.latex_output ? `${fmtInt(w.latex_output.length)} chars` : '—') + kv('Title', (fp.title || (w.selected_topic || {}).title || '—').slice(0, 80));
  const btn = $('approveBtn');
  if (noRun) { btn.style.display = 'none'; }
  else if (rel.status === 'ready' || w.human_approved) { btn.style.display = ''; btn.disabled = true; btn.textContent = 'Approved ✓ (L3)'; }
  else if (rel.needs_approval) { btn.style.display = ''; btn.disabled = false; btn.textContent = 'Approve for submission (L3)'; }
  else { btn.style.display = ''; btn.disabled = true; btn.textContent = 'Approve for submission (L3)'; }
}
async function approveCurrentRun() {
  if (!dashboard.run_id) { toast('No run selected', 'bad'); return; }
  if (!await confirmAction('Approve release', `Approve run ${dashboard.run_id} for submission? This marks output/paper_output.tex as the publishable release.`)) return;
  try { await api('/api/release/approve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_id: dashboard.run_id }) }); toast('Approved for submission', 'ok'); lastPhase = null; refreshFull(); }
  catch (e) { toast(e.message, 'bad'); }
}

/* ---------- routing ---------- */
function renderRouting(w) {
  const cands = w.paper_candidates || [];
  $('routingTag').textContent = w.paper_type || 'auto';
  $('routingRows').innerHTML =
    kv('Mode', w.hunt_mode || 'try_and_report') +
    kv('Stage', (w.routing_provenance || {}).stage || 'intake') +
    (cands.length ? cands.slice(0, 5).map((c) => `<div class="row"><div><strong>${esc(c.type)}</strong><small>${esc(c.why || '')}</small></div><span class="pill">${esc(String(c.confidence ?? ''))}</span></div>`).join('') : '<div class="empty">Candidates appear for auto runs after intake.</div>');
}

/* ---------- topics ---------- */
let topicFilter = '';
function renderTopics(w) {
  const topics = w.topics || [];
  const sel = (w.selected_topic || {}).title || '';
  $('topicCount').textContent = `${topics.length} topic${topics.length === 1 ? '' : 's'}`;
  const f = topicFilter.trim().toLowerCase();
  const list = f ? topics.filter((t) => JSON.stringify(t).toLowerCase().includes(f)) : topics;
  if (!list.length) { $('topicCards').innerHTML = `<div class="empty">No topics yet — start a run or wait for discovery.${f ? ' (filter active)' : ''}</div>`; return; }
  $('topicCards').innerHTML = list.slice(0, 12).map((t) => {
    const score = debatedScore(t.title, w);
    const pct = Math.max(0, Math.min(100, Math.round((num(score, 0) / 10) * 100)));
    return `<div class="topic-card${t.title === sel ? ' sel' : ''}">
      <div class="topic-title">${esc(t.title || 'Untitled')}</div>
      <div class="topic-meta">${esc((t.hunt_mode || w.hunt_mode || ''))} · ${esc((t.fallback_provenance || t.plan_mode || ''))}</div>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <div class="topic-foot"><span>score ${score == null ? '—' : num(score).toFixed(1)}</span><span>${t.title === sel ? '★ selected' : esc(short(t.status || t.gap_type || 'candidate', 18))}</span></div>
    </div>`;
  }).join('');
}
function debatedScore(title, w) {
  const rs = (w.debate_results || []).filter((r) => (r.topic || r.title) === title);
  if (!rs.length) return null;
  return num(rs[rs.length - 1].score, null);
}

/* ---------- activity / preflight / snapshot ---------- */
function renderActivity(d) {
  const msgs = d.recent_messages || d.events_tail || [];
  $('activity').innerHTML = msgs.length ? msgs.slice(-14).reverse().map((m) => {
    const text = typeof m === 'string' ? m : (m.message_human || m.message || m.type || JSON.stringify(m).slice(0, 140));
    const time = (m.timestamp || m.ts || '').slice(11, 19);
    return `<div class="event">${time ? `<time>${esc(time)}</time>` : ''}${esc(String(text)).slice(0, 200)}</div>`;
  }).join('') : '<div class="empty">No activity yet.</div>';
  const sp = scratch.scratchpad || scratch.lines || [];
  $('scratchpad').textContent = Array.isArray(sp) ? sp.slice(-40).map((l) => (typeof l === 'string' ? l : JSON.stringify(l))).join('\n') : String(sp || 'No notes yet.');
  const cr = d.cross_run || {};
  $('crossRun').innerHTML = ['rejections', 'pivots', 'runs'].map((k) => {
    const arr = cr[k] || [];
    return `<div class="row"><div><strong>${esc(k)}</strong><small>${esc((arr[0] && (arr[0].item || arr[0].kind || '')) || '—')}</small></div><span class="pill">${arr.length}</span></div>`;
  }).join('');
  const w = ws();
  const deb = (w.debate_results || []).slice(-3).reverse();
  $('debateSnap').innerHTML = deb.length ? deb.map((r) => `<div class="row"><div><strong>${esc(r.topic || 'topic')}</strong><small>${esc(r.moderator_decision || r.decision || '')} · ${esc((r.reasoning || '').slice(0, 110))}</small></div><span class="pill">${esc(String(r.score ?? '—'))}</span></div>`).join('')
    : `<div class="row"><div><strong>Supervisor</strong><small>${esc(Object.entries(w.supervisor_scores || {}).slice(0, 3).map(([k, v]) => `${k}:${v}`).join(' · ') || 'no scores yet')}</small></div></div>`;
  const evs = d.events_tail || [];
  $('eventStream').innerHTML = evs.length ? evs.slice(-30).reverse().map((e) => `<div class="event"><time>${esc((e.timestamp || '').slice(11, 19))}</time>${esc(e.message_human || e.type || '')} <span style="color:var(--faint)">${esc(e.phase || '')}</span></div>`).join('') : '<div class="empty">No events yet.</div>';
}
function renderPreflight(w) {
  const pre = (w.intake_report || {}).preflight || {};
  const rows = [];
  if (pre.mini_screen) rows.push(kv('Mini screen', pre.mini_screen.blocked ? 'blocked' : 'pass' + (pre.mini_screen.degraded ? ' (degraded)' : '')));
  if (w.intake_report) rows.push(kv('Goal', String(w.intake_report.goal || '—').slice(0, 90)));
  if (!rows.length) rows.push('<div class="empty">Pre-flight runs before spend — submit a run to populate.</div>');
  $('preflightRows').innerHTML = rows.join('');
  $('preflightTag').textContent = rows.length > 1 || pre.mini_screen ? 'checked' : 'not checked';
}
function renderSnapshot(w) {
  const eg = w.evidence_gate || {};
  $('snapshotRows').innerHTML =
    kv('Evidence gate', `${eg.reason_code || '—'}${eg.terminal ? ' (terminal)' : ''}`) +
    kv('Terminal', w.terminal_error ? String(w.terminal_error).slice(0, 90) : '—') +
    kv('Iteration', w.iteration ?? 0);
}

/* ---------- experiments ---------- */
function expList(w) {
  const plan = w.plan || {}, outs = w.engineer_outputs || {}, reps = w.analysis_reports || {};
  return (plan.experiments || []).map((e) => {
    const name = e.name || 'unnamed';
    const out = outs[name] || {};
    const rep = reps[name] || {};
    const raw = out.raw_results_path || (out.results || {}).path || '';
    return { name, planned: true, executed: out.success === true, analyzed: !!rep.passed || !!Object.keys(rep).length, artifact: raw ? short(raw, 34) : (out.success === true ? 'metrics ✓' : '—'), detail: { plan: e, out, rep } };
  });
}
function renderExperiments(w) {
  const list = expList(w).filter((e) => !topicFilter || JSON.stringify(e).toLowerCase().includes(topicFilter.toLowerCase()));
  $('expBoardTag').textContent = `${list.length} listed`;
  $('expBoard').innerHTML = list.length ? list.map((e) => `<div class="exp-row"><div><div class="name">${esc(e.name)}</div><div class="meta">${esc((e.detail.plan || {}).type || '')}</div></div>
    <span class="pill yes">listed</span><span class="pill ${e.executed ? 'yes' : 'no'}">${e.executed ? 'ran ✓' : 'pending'}</span>
    <span class="pill ${e.analyzed ? 'yes' : 'no'}">${e.analyzed ? 'done' : '—'}</span><span class="meta mono">${esc(e.artifact)}</span></div>`).join('')
    : '<div class="empty">No experiments planned yet — the planner populates this after debate.</div>';
  $('expDetailGrid').innerHTML = list.slice(0, 4).map((e) => {
    const m = (e.detail.out.aggregate_metrics || (e.detail.out.results || {}).metrics || {});
    const keys = Object.keys(m).slice(0, 6);
    return `<article class="card span-6"><div class="card-head"><h3>${esc(e.name)}</h3><span class="tag">${e.executed ? 'executed' : 'planned'}</span></div>
      ${keys.length ? keys.map((k) => kv(k, String(m[k]).slice(0, 60))).join('') : '<div class="empty">No metrics yet.</div>'}</article>`;
  }).join('');
}

/* ---------- evidence ---------- */
function renderEvidence(d, w) {
  const panes = {
    experiments: w.engineer_outputs || {}, corpus: w.corpus || {}, response: (w.data_artifacts || {}), case: w.instrument || {}, thesis: w.thesis || {},
  };
  const renderObj = (o) => {
    if (o == null) return '<div class="empty">No data for this tab.</div>';
    if (typeof o !== 'object') return `<div class="mono">${esc(String(o)).slice(0, 600)}</div>`;
    const keys = Object.keys(o);
    if (!keys.length) return '<div class="empty">No data for this tab.</div>';
    return keys.slice(0, 8).map((k) => kv(k, JSON.stringify(o[k]).slice(0, 140))).join('');
  };
  Object.keys(panes).forEach(() => {});
  $('evPane-experiments').innerHTML = renderObj(panes.experiments);
  $('evPane-corpus').innerHTML = renderObj(panes.corpus);
  $('evPane-response').innerHTML = renderObj(panes.response);
  $('evPane-case').innerHTML = renderObj(panes.case);
  $('evPane-thesis').innerHTML = renderObj(panes.thesis);
  const findings = w.verification_findings || [];
  $('findings').innerHTML = findings.length ? findings.slice(0, 8).map((f) => `<div class="finding${f.blocking ? '' : ' ok'}"><strong>${esc(f.check || f.kind || 'finding')}</strong><div>${esc((f.message || f.note || '').slice(0, 220))}</div></div>`).join('') : '<div class="empty">No findings — independent validation has not reported.</div>';
  const da = w.data_artifacts || {};
  $('datasetCards').innerHTML = Object.keys(da).length ? Object.entries(da).slice(0, 6).map(([k, v]) => kv(k, JSON.stringify(v).slice(0, 120))).join('') : '<div class="empty">No dataset artifacts.</div>';
  const eo = w.engineer_outputs || {};
  $('execCards').innerHTML = Object.keys(eo).length ? Object.entries(eo).slice(0, 6).map(([k, v]) => kv(k, v.success ? 'success ✓' : ('failed: ' + String(v.error || '').slice(0, 60)))).join('') : '<div class="empty">No execution outputs.</div>';
  const claims = d.evidence_trace || [];
  $('claims').innerHTML = claims.length ? `<table><thead><tr><th>Claim</th><th>Status</th><th>Source</th></tr></thead><tbody>${claims.slice(0, 20).map((c) => `<tr><td>${esc((c.claim || c.text || JSON.stringify(c)).slice(0, 120))}</td><td>${esc(c.status || '')}</td><td>${esc(c.source || c.phase || '')}</td></tr>`).join('')}</tbody></table>` : '<div class="empty">Ledger claims appear here once recorded.</div>';
  const arts = d.artifacts || {};
  $('artifactTable').innerHTML = Object.keys(arts).length ? Object.entries(arts).slice(0, 8).map(([k, v]) => kv(k, JSON.stringify(v).slice(0, 120))).join('') : '<div class="empty">No staged artifacts.</div>';
}

/* ---------- paper ---------- */
function renderPaper(w) {
  const sections = w.draft_sections || {};
  const names = Object.keys(sections);
  const scores = w.supervisor_scores || {};
  $('paperScores').innerHTML = names.length ? names.map((n) => `<div class="chip"><span class="n">${esc(String(scores[n] ?? '—'))}</span><span class="l">${esc(n)}</span></div>`).join('') : '<div class="empty">No sections drafted yet.</div>';
  $('tocList').innerHTML = names.length ? names.map((n, i) => `<a class="toc-item${i === 0 ? ' active' : ''}" data-sec="${esc(n)}">${esc(n)}</a>`).join('') : '<div class="empty">Sections appear here.</div>';
  document.querySelectorAll('.toc-item').forEach((a) => { a.onclick = () => { document.querySelectorAll('.toc-item').forEach((x) => x.classList.remove('active')); a.classList.add('active'); const el = $(`sec-${CSS.escape(a.dataset.sec)}`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }; });
  const f = topicFilter.trim().toLowerCase();
  const shown = f ? names.filter((n) => (n + sections[n]).toLowerCase().includes(f)) : names;
  $('paperContent').innerHTML = shown.length ? shown.map((n) => `<article class="paper" id="sec-${esc(n)}"><h2>${esc(n)}</h2><p>${esc(String(sections[n] || '').slice(0, 4000)).replaceAll('\n', '<br>')}</p></article>`).join('<div style="height:14px"></div>') : '<div class="empty">Manuscript sections will render here once writing completes.</div>';
}

/* ---------- agents (shape: {capabilities: {Name: {agent, role, allowed_capabilities,
   forbidden_capabilities, input_artifact_types, output_artifact_types, can_mutate}}}) ---------- */
let agentsCache = null;
const AGENT_FAMILIES = [
  ['Discovery', ['TopicHunterAgent']],
  ['Debate', ['HypothesisDebateSystem', 'ProposerAgent', 'ChallengerAgent', 'ModeratorAgent']],
  ['Planning', ['PlannerAgent']],
  ['Data', ['DataAgent']],
  ['Execution', ['EngineerAgent', 'ExecutionAgent', 'AnalysisAgent', 'VerificationAgent']],
  ['Evidence', ['ReviewAgent', 'SurveyAgent', 'SurveyAnalysisAgent', 'CaseAgent', 'CaseProtocolAgent', 'CausalAgent', 'CausalAuditAgent', 'ThesisAgent']],
  ['Writing', ['WriterAgent']],
  ['Oversight', ['SupervisorAgent', 'MetaAgent', 'EditorAgent']],
];
function agentFamilyOf(name) {
  for (const [fam, names] of AGENT_FAMILIES) if (names.includes(name)) return fam;
  return 'Other';
}
async function renderAgents() {
  const q = (($('agentSearch') || {}).value || '').toLowerCase();
  if (!agentsCache) { try { agentsCache = await api('/api/capabilities'); } catch { agentsCache = {}; } }
  const manifests = (agentsCache && agentsCache.capabilities) || agentsCache.roles || agentsCache.agents || {};
  const list = Object.entries(manifests).map(([name, m]) => ({ name, ...(typeof m === 'object' && m ? m : {}) }));
  const filtered = q ? list.filter((a) => JSON.stringify(a).toLowerCase().includes(q)) : list;
  $('agentCount').textContent = `${filtered.length} roles`;
  if (!filtered.length) { $('agentGrid').innerHTML = '<div class="empty">No agent roles reported.</div>'; return; }
  const order = [...AGENT_FAMILIES.map((f) => f[0]), 'Other'];
  const byFam = {};
  filtered.forEach((a) => { const f = agentFamilyOf(a.name || a.agent); (byFam[f] = byFam[f] || []).push(a); });
  $('agentGrid').innerHTML = order.filter((f) => byFam[f]).map((fam) =>
    `<div class="agent-family"><span class="agent-family-name">${esc(fam)} · ${byFam[fam].length}</span></div>` +
    byFam[fam].map((a) => {
      const name = a.name || a.agent || 'agent';
      const allowed = a.allowed_capabilities || a.capabilities || a.caps || a.tools || [];
      const forbidden = a.forbidden_capabilities || [];
      const inputs = a.input_artifact_types || [];
      const outputs = a.output_artifact_types || [];
      const mutates = a.can_mutate || [];
      const cap = (c, cls) => `<span class="cap${cls ? ' ' + cls : ''}">${esc(typeof c === 'string' ? c : JSON.stringify(c))}</span>`;
      return `<article class="card span-4 agent-card"><h4>${esc(name)}</h4><div class="role">${esc(a.role || a.description || '')}</div>
        <div class="agent-meta"><span class="tag">${esc(String((allowed || []).length))} allowed</span><span class="tag">${esc(String((forbidden || []).length))} denied</span></div>
        <div class="cap-list">${(Array.isArray(allowed) ? allowed : []).map((c) => cap(c)).join('') || '<span class="cap">—</span>'}</div>
        ${forbidden.length ? `<div class="cap-list">${forbidden.map((c) => cap(c, 'blocked')).join('')}</div>` : ''}
        ${(inputs.length || outputs.length) ? `<div class="cap-list">${inputs.map((c) => cap('in:' + c, 'io-in')).join('')}${outputs.map((c) => cap('out:' + c, 'io-out')).join('')}</div>` : ''}
        ${mutates.length ? `<div class="note" style="margin-top:8px">mutates: ${esc(mutates.join(', '))}</div>` : ''}</article>`;
    }).join('')
  ).join('');
}

/* ---------- runs ---------- */
let runFilter = 'all', runQuery = '';
async function loadRuns() {
  let data = {};
  try { data = await api('/api/runs'); } catch (e) { $('runTable').innerHTML = `<div class="empty">Could not load runs: ${esc(e.message)}</div>`; return; }
  const runs = data.runs || data || [];
  const list = Array.isArray(runs) ? runs : [];
  const q = runQuery.toLowerCase();
  const shown = list.filter((r) => (runFilter === 'all' || (r.mode || r.type) === runFilter || runFilter === 'full_research') && (!q || JSON.stringify(r).toLowerCase().includes(q)));
  $('runTable').innerHTML = shown.length ? shown.slice(0, 40).map((r) => `<div class="run-row">
      <span class="run-id-mono">${esc(r.run_id || '')}</span>
      <span>${esc(r.status || r.phase || '')} <span style="color:var(--faint)">· ${esc(r.mode || '')}</span></span>
      <span class="meta">${esc((r.updated_at || r.created_at || '').slice(0, 16))}</span>
      <span><button data-act="open" data-run="${esc(r.run_id || '')}">Open</button></span>
      <span style="display:flex;gap:6px"><button data-act="resume" data-run="${esc(r.run_id || '')}">Resume</button><button data-act="rerun" data-run="${esc(r.run_id || '')}">Re-run</button><button class="danger" data-act="del" data-run="${esc(r.run_id || '')}">✕</button></span>
    </div>`).join('') : '<div class="empty">No runs in the ledger yet.</div>';
  document.querySelectorAll('#runTable [data-act]').forEach((b) => { b.onclick = () => runRowAction(b.dataset.act, b.dataset.run); });
}
async function runRowAction(act, id) {
  if (act === 'open') { selectedRunId = id; await refreshFull(); showView('overview'); toast('Opened run ' + id, 'ok'); }
  if (act === 'resume') { try { await api(`/api/run/resume/${encodeURIComponent(id)}`, { method: 'POST' }); selectedRunId = id; toast('Resumed ' + id, 'ok'); } catch (e) { toast(e.message, 'bad'); } }
  if (act === 'rerun') { if (!await confirmAction('Re-run', `Start ${id} from scratch?`)) return; try { await api(`/api/run/rerun/${encodeURIComponent(id)}`, { method: 'POST' }); selectedRunId = id; toast('Re-running ' + id, 'ok'); } catch (e) { toast(e.message, 'bad'); } }
  if (act === 'del') { if (!await confirmAction('Delete run', `Delete ${id} from the ledger?`)) return; try { await api(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' }); loadRuns(); toast('Deleted', 'ok'); } catch (e) { toast(e.message, 'bad'); } }
}
function renderRunsBadge() {}

/* ---------- settings (shape pinned by tests/test_settings_surface.py:
   each entry is ['id','Title','type','Group','help...'];
   lowercase ids resolve via /api/config, UPPER via /api/keys) ---------- */
const settingDefs=[
  ['llm_provider', 'LLM provider', 'provider', 'LLM', 'Chat backend: gemini, openai, or openai_compatible (any OpenAI-style endpoint: OpenRouter, Groq, local Ollama/vLLM). Embeddings ALWAYS stay on Gemini regardless of this choice.'],
  ['gemini_model', 'Gemini / Gemma model', 'text', 'LLM', 'Chat model id when provider is gemini. Default: gemma-4-26b-a4b-it. Other Gemini and Gemma model IDs remain supported. Must be a valid model id with no spaces.'],
  ['gemma_thinking_level', 'Gemma 4 thinking', 'select', 'LLM', 'Gemma 4 only: high enables its maximum supported reasoning mode; minimal disables it. Ignored by all other models.'],
  ['gemini_embedding_model', 'Gemini embedding model', 'text', 'LLM', 'LOCKED: gemini-embedding-2 at 3072 dims. The stored FAISS index is 3072-wide — changing this corrupts similarity search. Do not touch.'],
  ['openai_model', 'OpenAI model', 'text', 'LLM', 'Chat model id when provider is openai or openai_compatible. E.g. gpt-4o-mini, or a vendor path like MiniMaxAI/MiniMax-M2.7.'],
  ['openai_embedding_model', 'OpenAI embedding model', 'text', 'LLM', 'Display only — NEVER used for embeddings (Gemini-only policy). Leave as-is.'],
  ['openai_base_url', 'OpenAI-compatible base URL', 'text', 'LLM', 'Endpoint for openai_compatible, e.g. https://api.openai.com/v1 or http://127.0.0.1:11434/v1 for local Ollama. Required when provider is openai_compatible.'],
  ['llm_model_cheap', 'Cheap model override', 'text', 'LLM', 'Optional: model id for cheap-tier calls (lookups, screens). Blank = provider default. Set to a -lite/-mini id to cut spend.'],
  ['llm_model_strong', 'Strong model override', 'text', 'LLM', 'Optional: model id for debate, writing, and verification. Blank = provider default. Set to a pro-grade id for harder work.'],
  ['llm_model_judge', 'Judge model override', 'text', 'LLM', 'Optional: model id for referee/consistency verdicts. Blank = provider default.'],
  ['llm_model_fallback', 'Fallback model', 'text', 'LLM', 'Model id tried when the primary provider has connection errors (with LLM_FALLBACK_PROVIDER). Blank = no model-level fallback.'],
  ['llm_fallback_provider', 'Fallback provider', 'text', 'LLM', 'Backup provider on connection errors: gemini or openai_compatible (blank = none). Its key must be set or startup validation fails.'],
  ['llm_fallback_model', 'Fallback model id', 'text', 'LLM', 'Model id used on the fallback provider. Blank = provider default model.'],
  ['llm_timeout_sec', 'LLM timeout (seconds)', 'number', 'LLM', 'Per-request HTTP timeout. MUST exceed your slowest response — slow reasoning endpoints need 1800. Too low cuts healthy streams off mid-answer.'],
  ['llm_max_output_tokens', 'Max output tokens ceiling', 'number', 'LLM', 'Truncation-escalation ceiling (default 65536). Persona prompts burn budget on reasoning; below 32768 risks finish_reason=length empties.'],
  ['llm_persona_max_tokens', 'Persona first-attempt tokens', 'number', 'LLM', 'First-attempt budget for persona gap generation (default 16384, escalates to ceiling above).'],
  ['llm_price_table_json', 'Price table override (JSON)', 'text', 'LLM', 'Optional JSON overriding per-million-token prices for spend accounting. Blank = built-in estimates.'],
  ['ensemble_judge_models', 'Ensemble judge models', 'text', 'LLM', 'Optional comma-separated extra judge models for ensemble verdicts. Blank = single judge.'],
  ['research_domain', 'Research domain', 'text', 'Research', 'Default domain for topic discovery, e.g. computer_science, machine_learning, biology. Can be overridden per run in the Start dialog.'],
  ['openalex_email', 'OpenAlex contact email', 'email', 'Research', 'REQUIRED: a real email for the OpenAlex polite pool (rate limits). Placeholder addresses are rejected at startup.'],
  ['supervisor_threshold', 'Supervisor threshold', 'number', 'Research', 'Mean section score needed to pass supervision (default 8.5). Lower = easier pass, weaker bar.'],
  ['debate_pass_threshold', 'Debate pass threshold', 'number', 'Research', 'Mean judge score for a hypothesis to survive debate (default 7.0). Unresolved severity-4+ objections still block regardless.'],
  ['debate_min_rounds', 'Debate minimum rounds', 'number', 'Research', 'Floor of challenger/proposer rounds per topic (default 2). Raises rigor and cost.'],
  ['debate_max_rounds', 'Debate maximum rounds', 'number', 'Research', 'Ceiling of debate rounds (default 5). High disagreement extends the debate up to this cap.'],
  ['novelty_similarity_reject', 'Novelty rejection threshold', 'number', 'Research', 'Cosine similarity above which a topic is rejected as already-done (default 0.92). Lower = stricter novelty bar.'],
  ['experiment_seeds', 'Experiment seeds', 'number', 'Research', 'Multi-seed repetitions per experiment (default 3). More seeds = tighter confidence intervals, more compute.'],
  ['experiment_branch_count', 'Experiment branch count', 'number', 'Research', 'Candidate approaches probed cheaply before the winner runs multi-seed (default 3).'],
  ['max_iterations', 'Maximum iterations', 'number', 'Research', 'Meta-evaluation continue-loop cap (default 10).'],
  ['engineer_max_attempts', 'Engineer attempts', 'number', 'Harness', 'REFINE/PIVOT attempts per experiment before giving up (default 4).'],
  ['typed_graph_enabled', 'Typed recipes (master switch)', 'checkbox', 'Harness', 'ON = each run follows its paper-type recipe (10 types, per-type gates). OFF = legacy fixed pipeline. Keep ON.'],
  ['paper_type_default', 'Default paper type', 'text', 'Harness', 'Used when intake cannot decide a type. One of the 10 paper type ids.'],
  ['parallel_phases_enabled', 'Parallel phases', 'checkbox', 'Harness', 'Experimental opt-in fan-out. Keep OFF — execution stays sequential and deterministic.'],
  ['run_max_llm_calls', 'Run LLM call cap', 'number', 'Harness', 'Backstop on total LLM calls per run (default 2000).'],
  ['run_max_tokens', 'Run token cap', 'number', 'Harness', 'Backstop on total tokens per run (default 8000000).'],
  ['run_max_usd', 'Run spend cap (USD)', 'number', 'Harness', 'Backstop on estimated spend per run (default 500).'],
  ['run_wall_clock_sec', 'Run wall-clock cap (s)', 'number', 'Harness', 'Hard deadline per run in seconds (default 21600 = 6h).'],
  ['engineering_wall_clock_sec', 'Engineering wall-clock cap (s)', 'number', 'Harness', 'Total engineering-phase budget across all experiments (default 7200).'],
  ['topic_seed_timeout_sec', 'Seed timeout (s)', 'number', 'Harness', 'Budget per discovery seed (default 5400). Too low starves slow endpoints.'],
  ['topic_discovery_timeout_sec', 'Discovery timeout (s)', 'number', 'Harness', 'Whole-discovery budget (default 21600). Auto-scales to fit seed count.'],
  ['topic_fallback_enabled', 'Topic fallback', 'checkbox', 'Harness', 'ON = keep a bounded, explicitly labelled exploratory candidate instead of discarding an investigation when a gate/provider is unavailable.'],
  ['topic_fallback_max_candidates', 'Max fallback candidates', 'number', 'Harness', 'Cap on fallback candidates kept per run (default 2).'],
  ['seed_sequential', 'Sequential seeds', 'checkbox', 'Harness', 'ON = process discovery seeds one at a time (gateway-paced). Turning off risks API bans.'],
  ['persona_sequential', 'Sequential personas', 'checkbox', 'Harness', 'ON = skeptic/practitioner persona calls run in sequence inside each seed.'],
  ['persona_count', 'Persona count', 'number', 'Harness', 'Personas per seed (default 2). More personas = broader critique, more LLM calls.'],
  ['llm_budget_per_seed', 'LLM budget per seed', 'number', 'Harness', 'Max LLM calls per discovery seed (default 30, split generation/gate-chain pools).'],
  ['novelty_max_abstracts', 'Novelty abstract cap', 'number', 'Harness', 'Abstracts embedded once per seed and reused across gap evaluations (default 20).'],
  ['llm_seed_generation_enabled', 'LLM seed generation', 'checkbox', 'Harness', 'ON = mint diverse technical seeds from cross-run lessons; falls back to template seeds on failure.'],
  ['sandbox_backend', 'Sandbox backend', 'text', 'Harness', 'docker or ast. Docker = trust boundary; ast = in-process policy filter fallback.'],
  ['arxiv_enabled', 'arXiv source', 'checkbox', 'Harness', 'ON = search arXiv (official client, 1 req/3s).'],
  ['openalex_enabled', 'OpenAlex source', 'checkbox', 'Harness', 'ON = search OpenAlex (polite pool via your email).'],
  ['semantic_scholar_enabled', 'Semantic Scholar source', 'checkbox', 'Harness', 'ON = search Semantic Scholar (bulk + citation graph, 1 req/s). OFF skips all S2 calls.'],
  ['unpaywall_enabled', 'Unpaywall source', 'checkbox', 'Harness', 'ON = resolve open-access URLs via Unpaywall (free, 1 req/s).'],
  ['pmc_enabled', 'PMC source', 'checkbox', 'Harness', 'ON = PubMed Central OA full-text links (biomedical).'],
  ['doaj_guard_enabled', 'DOAJ guard', 'checkbox', 'Harness', 'ON = warn when an OA claim comes from a non-DOAJ-listed journal.'],
  ['openreview_enabled', 'OpenReview source', 'checkbox', 'Harness', 'ON = ICLR/NeurIPS review data for calibration and gap signals.'],
  ['openreview_username', 'OpenReview username', 'text', 'Harness', 'OpenReview account email/username for review-queue access.'],
  ['huggingface_max_download_mb', 'HF download cap (MB)', 'number', 'Harness', 'Max dataset download size from HuggingFace Hub (default 100).'],
  ['allow_synthetic_respondents', 'Allow synthetic respondents', 'checkbox', 'Harness', 'LOCKED OFF — fabricating respondents is a release-blocker. Never enable.'],
  ['debug_mode', 'Debug mode', 'checkbox', 'Runtime', 'ON = verbose per-run logs. Turn off to slim logs.'],
  ['log_level', 'Log level', 'text', 'Runtime', 'Global log verbosity: DEBUG, INFO, WARNING, or ERROR.'],
  ['memory_size', 'Memory size', 'number', 'Runtime', 'Cross-run lesson store capacity (default 10000).'],
  ['sandbox_timeout_sec', 'Sandbox timeout (seconds)', 'number', 'Runtime', 'Soft timeout per sandboxed execution (default 600).'],
  ['sandbox_max_output_bytes', 'Sandbox max output bytes', 'number', 'Runtime', 'Output cap per execution (default 1048576 = 1MB).'],
  ['web_host', 'Web host', 'text', 'Runtime', 'Dashboard bind address (default 127.0.0.1 = local only).'],
  ['web_port', 'Web port', 'number', 'Runtime', 'Dashboard port (default 8765).'],
  ['vector_db_path', 'Vector database path', 'text', 'Storage', 'FAISS similarity index location (default ./memory/vector_db). 3072-wide Gemini vectors.'],
  ['cross_run_memory_path', 'Cross-run memory path', 'text', 'Storage', 'Science-lessons JSONL shared across runs (default ./memory/cross_run.jsonl).'],
  ['run_log_path', 'Run scratchpad path', 'text', 'Storage', 'Raw agent content per run (default ./output/run_scratchpad.jsonl).'],
  ['run_events_path', 'Run events path', 'text', 'Storage', 'Machine event stream (default ./output/run_events.jsonl).'],
  ['elo_ratings_path', 'Elo ratings path', 'text', 'Storage', 'Topic/strategy Elo ratings (default ./memory/elo_ratings.json).'],
  ['checkpoint_path', 'Checkpoint path', 'text', 'Storage', 'LangGraph SqliteSaver for true resume (default ./memory/checkpoints.sqlite).'],
  ['research_db_path', 'Research ledger path', 'text', 'Storage', 'Durable runs/events/claims/artifacts DB (default ./memory/research_ledger.sqlite).'],
  ['output_dir', 'Output directory', 'text', 'Storage', 'Root for papers, bundles, and caches (default ./output).'],
  ['draft_versions_dir', 'Draft versions directory', 'text', 'Storage', 'Versioned manuscript drafts (default ./output/draft_versions).'],
  ['debate_log_path', 'Debate log path', 'text', 'Storage', 'Tournament/round transcripts (default ./output/debate_log.json).'],
  ['feedback_log_path', 'Feedback log path', 'text', 'Storage', 'Supervisor/meta feedback trail (default ./output/feedback_log.json).'],
  ['raw_results_dir', 'Raw results directory', 'text', 'Storage', 'Multi-seed raw JSON per experiment (default ./output/raw_results).'],
  ['companion_repo_dir', 'Companion repo directory', 'text', 'Storage', 'Reproducibility bundle root (default ./output/companion_repo).'],
  ['keys_store_path', 'Keys store path', 'text', 'Storage', 'UI-entered secrets location (default ./memory/keys.json). Local only, gitignored.'],
  ['GOOGLE_API_KEY', 'Google API key', 'password', 'Credentials', 'Gemini chat + embeddings key. Required when provider is gemini. Stored masked; empty fields never overwrite.'],
  ['OPENAI_API_KEY', 'OpenAI API key', 'password', 'Credentials', 'Key for openai/openai_compatible. Required for those providers.'],
  ['SEMANTIC_SCHOLAR_API_KEY', 'Semantic Scholar API key', 'password', 'Credentials', 'Optional: raises the Semantic Scholar rate limit (works without, slower).'],
  ['HUGGINGFACE_TOKEN', 'HuggingFace token', 'password', 'Credentials', 'Optional: raises Hub quotas for public dataset manifests.'],
  ['OPENREVIEW_USERNAME', 'OpenReview username', 'text', 'Credentials', 'OpenReview account for review-queue access. Username is shown in clear; the password below is always masked.'],
  ['OPENREVIEW_PASSWORD', 'OpenReview password', 'password', 'Credentials', 'OpenReview password. Stored with your keys, never logged, never rendered back.'],
];
const SETTING_GROUPS = ['LLM', 'Research', 'Harness', 'Runtime', 'Storage', 'Credentials'];
async function renderSettings() {
  if (!Object.keys(settingsCache).length) { try { settingsCache = await api('/api/config'); } catch { settingsCache = {}; } }
  if (!Object.keys(keysCache).length) { try { keysCache = await api('/api/keys'); } catch { keysCache = {}; } }
  const cfg = settingsCache, keys = (keysCache.keys || {});
  $('settingsTabs').innerHTML = SETTING_GROUPS.map((g) => `<button data-sg="${g}" class="${g === activeSettingsGroup ? 'primary' : ''}">${g}</button>`).join('');
  document.querySelectorAll('#settingsTabs [data-sg]').forEach((b) => { b.onclick = () => { activeSettingsGroup = b.dataset.sg; renderSettings(); }; });
  const defs = settingDefs.filter((d) => d[3] === activeSettingsGroup);
  const toField = (upper) => upper;
  $('settingsForm').innerHTML = defs.map(([key, label, type, _g, help]) => {
    let val = cfg[key.toLowerCase()] ?? keys[key] ?? '';
    if (type === 'password') val = '';
    const helpHtml = help ? `<span class="setting-help" tabindex="0" data-tip="${esc(help)}">?</span>` : '';
    if (type === 'checkbox') return `<div><label><input type="checkbox" data-key="${esc(key)}" ${val === true || val === 'true' ? 'checked' : ''} style="width:auto;margin-right:8px">${esc(label)}${helpHtml}</label></div>`;
    if (type === 'provider') return `<div><label>${esc(label)}${helpHtml}</label><select data-key="${esc(key)}"><option ${val === 'gemini' ? 'selected' : ''}>gemini</option><option value="openai" ${val === 'openai' ? 'selected' : ''}>openai</option><option value="openai_compatible" ${val === 'openai_compatible' ? 'selected' : ''}>openai_compatible</option></select></div>`;
    if (key === 'gemma_thinking_level') return `<div><label>${esc(label)}${helpHtml}</label><select data-key="${esc(key)}"><option value="high" ${val === 'high' ? 'selected' : ''}>high (maximum reasoning)</option><option value="minimal" ${val === 'minimal' ? 'selected' : ''}>minimal (off)</option></select></div>`;
    return `<div><label>${esc(label)}${helpHtml}</label><input data-key="${esc(key)}" type="${esc(type)}" value="${esc(type === 'password' ? '' : val)}" placeholder="${type === 'password' ? 'leave blank to keep saved' : ''}"></div>`;
  }).join('');
  renderPresets();
}

/* ---------- presets ---------- */
const PRESETS_KEY = 'scholargraph-presets-v2';
const getPresets = () => { try { return JSON.parse(localStorage.getItem(PRESETS_KEY) || '[]'); } catch { return []; } };
function renderPresets() {
  const presets = getPresets();
  $('presetList').innerHTML = presets.map((p) => `<div class="preset-chip" data-preset="${esc(p.name)}">${esc(p.name)}<span data-del="${esc(p.name)}" style="color:var(--red);cursor:pointer">×</span></div>`).join('') || '<span class="note">none saved</span>';
  document.querySelectorAll('#presetList [data-preset]').forEach((el) => {
    el.onclick = (e) => {
      const name = el.dataset.preset;
      if (e.target.dataset.del) { localStorage.setItem(PRESETS_KEY, JSON.stringify(getPresets().filter((p) => p.name !== name))); renderPresets(); return; }
      const p = getPresets().find((x) => x.name === name);
      if (p) { Object.entries(p.values).forEach(([k, v]) => { const inp = document.querySelector(`[data-key="${CSS.escape(k)}"]`); if (inp) { if (inp.type === 'checkbox') inp.checked = !!v; else inp.value = v; } }); toast('Preset applied: ' + name, 'ok'); }
    };
  });
}

/* ---------- start modal (defaults = live /api/config ≈ .env, editable per run) ---------- */
async function openModal() {
  try {
    settingsCache = await api('/api/config');
  } catch {}
  const c = settingsCache || {};
  if ($('intentDomain')) {
    // Per-run override: always start from the live env default, user can edit.
    $('intentDomain').value = c.research_domain || '';
    $('intentDomain').placeholder = c.research_domain ? `default from .env: ${c.research_domain} (editable)` : 'blank = .env RESEARCH_DOMAIN';
    $('intentDomain').title = 'Per-run override — blank uses .env RESEARCH_DOMAIN';
  }
  if ($('intentPaperType') && c.paper_type_default) {
    // 'auto' already resolves server-side to PAPER_TYPE_DEFAULT; surface it.
    const autoOpt = $('intentPaperType').querySelector('option[value="auto"]');
    if (autoOpt) autoOpt.textContent = `auto (recommended → ${c.paper_type_default})`;
  }
  $('intentModal').classList.add('show');
}
function closeModal() { $('intentModal').classList.remove('show'); }
async function uploadPacks() {
  const fd = new FormData();
  const map = [['packResponseTable', 'response_table'], ['packCodebook', 'codebook'], ['packCasePack', 'case_pack'], ['packCorpusOverride', 'corpus_override']];
  let n = 0;
  map.forEach(([id, field]) => { const el = $(id); if (el && el.files && el.files[0]) { fd.append(field, el.files[0]); n++; } });
  if (!n) { $('packStatus').textContent = 'Select a file first.'; return; }
  try { const r = await fetch('/api/packs', { method: 'POST', body: fd }).then((x) => x.json()); $('packStatus').textContent = 'Uploaded: ' + (r.pack_id || r.status || 'ok'); if (r.pack_id && !$('intentPackId').value) $('intentPackId').value = r.pack_id; }
  catch (e) { $('packStatus').textContent = 'Upload failed: ' + e.message; }
}
async function submitRun() {
  const err = $('intentError'); err.style.display = 'none';
  const numOrNull = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) && v > 0 ? v : null; };
  const body = {
    goal: $('intentGoal').value || null,
    domain: $('intentDomain').value || null,
    paper_type: $('intentPaperType').value || 'auto',
    hunt_mode: $('intentHuntMode').value || 'try_and_report',
    depth: $('intentDepth').value || 'Standard',
    pack_id: $('intentPackId').value || null,
    budget_usd: numOrNull('intentBudgetUsd'), budget_min: numOrNull('intentBudgetMin'),
    mode: 'full_research',
  };
  try {
    const res = await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    selectedRunId = res.run_id || null; closeModal(); toast('Run started: ' + (selectedRunId || ''), 'ok');
    lastPhase = null; refreshFull(); tick();
  } catch (e) { err.textContent = e.message; err.style.display = ''; }
}

/* ---------- actions ---------- */
async function saveSettings() {
  // Only the active tab is in the DOM — merge visible edits over the full
  // last-known config (UPPER keys, matching the backend) so saving never
  // drops other tabs' values and applies immediately, even mid-run.
  const merged = {};
  _copyMap(merged, settingsCache);
  _copyMap(merged, (keysCache || {}).keys, true);
  document.querySelectorAll('#settingsForm [data-key]').forEach((el) => {
    const k = String(el.dataset.key).toUpperCase();
    if (el.type === 'checkbox') merged[k] = el.checked;
    else if (el.value !== '') merged[k] = el.value;
  });
  try {
    const res = await api('/api/keys', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ keys: merged }) });
    $('providerStatus').textContent = 'Saved ✓ provider=' + (res.provider || '?'); $('providerStatus').className = 'tag good';
    try { settingsCache = await api('/api/config'); } catch { settingsCache = {}; }
    try { keysCache = await api('/api/keys'); } catch { keysCache = {}; }
    renderSettings();
    toast('Settings saved + applied', 'ok');
  } catch (e) { $('providerStatus').textContent = 'Save failed: ' + e.message; $('providerStatus').className = 'tag bad'; }
}
async function copyText(t, btn) {
  try { await navigator.clipboard.writeText(t); if (btn) { const o = btn.textContent; btn.textContent = 'copied ✓'; setTimeout(() => (btn.textContent = o), 1200); } toast('Copied', 'ok'); }
  catch { toast('Copy failed', 'bad'); }
}

/* ---------- wire up ---------- */
function bind() {
  document.querySelectorAll('.nav button').forEach((b) => { b.onclick = () => showView(b.dataset.view); });
  $('startButton').onclick = openModal;
  $('intentCancel').onclick = closeModal;
  $('intentModal').addEventListener('click', (e) => { if (e.target === $('intentModal')) closeModal(); });
  $('packUploadBtn').onclick = uploadPacks;
  $('intentSubmit').onclick = submitRun;
  $('trueResumeBtn').onclick = async () => { const id = $('resumeRunId').value.trim() || selectedRunId; if (!id) return toast('Enter a run id', 'bad'); try { await api(`/api/run/resume/${encodeURIComponent(id)}`, { method: 'POST' }); selectedRunId = id; closeModal(); toast('Resumed ' + id, 'ok'); } catch (e) { toast(e.message, 'bad'); } };
  $('rerunBtn').onclick = async () => { const id = $('resumeRunId').value.trim() || selectedRunId; if (!id) return toast('Enter a run id', 'bad'); try { await api(`/api/run/rerun/${encodeURIComponent(id)}`, { method: 'POST' }); selectedRunId = id; closeModal(); toast('Re-running ' + id, 'ok'); } catch (e) { toast(e.message, 'bad'); } };
  $('refreshButton').onclick = () => { lastPhase = null; refreshFull(); };
  $('refreshButtonRail').onclick = () => { lastPhase = null; refreshFull(); };
  $('darkModeToggle').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  $('copyRunId').onclick = (e) => copyText(dashboard.run_id || '', e.target);
  $('runId').onclick = (e) => copyText(dashboard.run_id || '', null);
  $('resetOutputsButton').onclick = async () => { if (!await confirmAction('Reset outputs', 'Delete generated outputs? This cannot be undone.')) return; try { await api('/api/data/reset/outputs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirmation: 'yes' }) }); toast('Outputs reset', 'ok'); refreshFull(); } catch (e) { toast(e.message, 'bad'); } };
  $('saveSettings').onclick = saveSettings;
  $('testProvider').onclick = async () => { try { $('providerStatus').textContent = 'Testing…'; const r = await api('/api/keys/test', { method: 'POST' }); $('providerStatus').textContent = 'LLM OK: ' + (r.response || 'connected'); $('providerStatus').className = 'tag good'; } catch (e) { $('providerStatus').textContent = 'Failed: ' + e.message; $('providerStatus').className = 'tag bad'; } };
  $('savePresetBtn').onclick = () => {
    const name = $('presetNameInput').value.trim(); if (!name) return toast('Name the preset first', 'bad');
    const values = {}; document.querySelectorAll('#settingsForm [data-key]').forEach((el) => { values[el.dataset.key] = el.type === 'checkbox' ? el.checked : el.value; });
    const all = getPresets().filter((p) => p.name !== name); all.push({ name, values });
    try { localStorage.setItem(PRESETS_KEY, JSON.stringify(all)); } catch {}
    $('presetNameInput').value = ''; renderPresets(); toast('Preset saved', 'ok');
  };
  $('approveBtn').onclick = approveCurrentRun;
  $('approveBtnGate').onclick = approveCurrentRun;
  $('rerunNoteBtn').onclick = async () => { const note = $('rerunNote').value.trim(); if (!dashboard.run_id) return toast('No run selected', 'bad'); try { await api(`/api/run/rerun/${encodeURIComponent(dashboard.run_id)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ note }) }); toast('Re-run requested', 'ok'); } catch (e) { toast(e.message, 'bad'); } };
  $('copyExpSummary').onclick = (e) => copyText($('expBoard').innerText || '', e.target);
  $('copyPaperBtn').onclick = (e) => copyText($('paperContent').innerText || '', e.target);
  $('agentSearch').oninput = () => renderAgents();
  $('globalSearch').oninput = (e) => { topicFilter = e.target.value || ''; renderTopics(ws()); renderExperiments(ws()); renderPaper(ws()); };
  $('runSearchInput').oninput = (e) => { runQuery = e.target.value || ''; loadRuns(); };
  document.querySelectorAll('.run-filter-btn').forEach((b) => { b.onclick = () => { document.querySelectorAll('.run-filter-btn').forEach((x) => x.classList.remove('active')); b.classList.add('active'); runFilter = b.dataset.filter; loadRuns(); }; });
  document.querySelectorAll('#evTabs [data-evtab]').forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll('#evTabs [data-evtab]').forEach((x) => x.classList.remove('primary'));
      b.classList.add('primary');
      ['experiments', 'corpus', 'response', 'case', 'thesis'].forEach((k) => { $('evPane-' + k).style.display = k === b.dataset.evtab ? '' : 'none'; });
    };
  });
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input,textarea,select')) { if (e.key === 'Escape') e.target.blur(); return; }
    const views = ['overview', 'experiments', 'evidence', 'agents', 'paper', 'activity', 'runs', 'settings'];
    if (e.key >= '1' && e.key <= '8') showView(views[+e.key - 1]);
    else if (e.key === '/') { e.preventDefault(); $('globalSearch').focus(); }
    else if (e.key === 'r') { lastPhase = null; refreshFull(); }
    else if (e.key === 'd') applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
    else if (e.key === 'Escape') closeModal();
  });
  setTimeout(() => $('kbdHint').classList.add('show'), 800);
  setTimeout(() => $('kbdHint').classList.remove('show'), 6000);
}

/* ---------- boot ---------- */
async function boot() {
  initTheme(); bind();
  if (new URLSearchParams(location.search).has('demo')) {
    await refreshFull();
    if ($('dbPath')) $('dbPath').textContent = 'design fixture · no API calls';
    return;
  }
  try {
    const [c, k] = await Promise.all([api('/api/config'), api('/api/keys').catch(() => ({}))]);
    settingsCache = c || {};
    keysCache = k || {};
    if ($('intentDomain')) $('intentDomain').value = c.research_domain || '';
    if ($('dbPath')) $('dbPath').textContent = `${c.llm_provider || '?'} · ${c.research_db_path || ''}`;
    const hasKeys = (k && (k.has_google_key || k.has_openai_key)) || c.llm_provider;
    $('onboardingCard').style.display = (!dashboard.run_id && !hasKeys) ? '' : 'none';
  } catch (e) { console.error(e); }
  await refreshFull();
  schedulePoll(3000);
}
document.addEventListener('DOMContentLoaded', boot);
