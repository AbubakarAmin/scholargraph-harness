/* ScholarGraph Chat — UI layer over shared harness state */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    app: $("#app"),
    messages: $("#messages"),
    hero: $("#hero"),
    composer: $("#composer"),
    form: $("#composerForm"),
    sendBtn: $("#sendBtn"),
    attachBtn: $("#attachBtn"),
    fileInput: $("#fileInput"),
    attachmentList: $("#attachmentList"),
    historyList: $("#historyList"),
    historyCount: $("#historyCount"),
    historyEmpty: $("#historyEmpty"),
    thinking: $("#thinking"),
    thinkingText: $("#thinkingText"),
    stageRail: $("#stageRail"),
    stages: $("#stages"),
    stageCount: $("#stageCount"),
    agentNowText: $("#agentNowText"),
    liveDot: $("#liveDot"),
    statusLabel: $("#statusLabel"),
    focusChip: $("#focusChip"),
    activeStrip: $("#activeStrip"),
    newChatBtn: $("#newChatBtn"),
    themeBtn: $("#themeBtn"),
    chatMain: $("#chatMain"),
  };

  const state = {
    sessionId: localStorage.getItem("sg_chat_session") || null,
    focusRunId: null,
    busy: false,
    harness: null,
    pollTimer: null,
    pendingFiles: [],
  };

  const DEFAULT_STAGES = [
    "intake",
    "topic_discovery",
    "hypothesis_debate",
    "planning",
    "engineering",
    "independent_validation",
    "writing_results",
    "editing",
  ];

  /* Theme sync with Control Deck (localStorage key: sg-theme) */
  function applyTheme(t) {
    const theme = t === "dark" ? "dark" : "light";
    if (theme === "dark") document.documentElement.dataset.theme = "dark";
    else document.documentElement.removeAttribute("data-theme");
    try {
      localStorage.setItem("sg-theme", theme);
    } catch {}
  }
  function initTheme() {
    let t = null;
    try {
      t = localStorage.getItem("sg-theme");
    } catch {}
    applyTheme(t === "dark" ? "dark" : "light");
  }
  function toggleTheme() {
    const cur = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    applyTheme(cur === "dark" ? "light" : "dark");
  }

  /* ---------- helpers ---------- */
  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function simpleMarkdown(text) {
    let s = escapeHtml(text || "");
    s = s.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    s = s.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    s = s.replace(/^\- (.+)$/gm, "<li>$1</li>");
    s = s.replace(/(<li>[\s\S]*?<\/li>)/g, (m) => `<ul>${m}</ul>`);
    s = s.replace(/\n\n+/g, "</p><p>");
    s = s.replace(/\n/g, "<br>");
    return `<p>${s}</p>`;
  }

  function shortPhase(name) {
    const map = {
      intake: "intake",
      topic_discovery: "topics",
      hypothesis_debate: "debate",
      planning: "plan",
      data_validation: "data",
      engineering: "eng",
      independent_validation: "validate",
      writing_narrative: "write",
      writing_results: "results",
      supervision: "review",
      editing: "edit",
    };
    return map[name] || String(name || "").replace(/_/g, " ").slice(0, 10);
  }

  function phaseLabel(name) {
    const labels = {
      intake: "Intake",
      topic_discovery: "Topic discovery",
      hypothesis_debate: "Hypothesis debate",
      planning: "Research planning",
      data_validation: "Data validation",
      engineering: "Experiment engineering",
      independent_validation: "Independent validation",
      writing_narrative: "Narrative writing",
      writing_results: "Results writing",
      supervision: "Research review",
      editing: "Final editing",
    };
    return labels[name] || String(name || "").replace(/_/g, " ");
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: opts.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail || res.statusText || "Request failed";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function setBusy(on, label) {
    state.busy = !!on;
    el.sendBtn.disabled = state.busy;
    el.attachBtn.disabled = state.busy;
    if (state.busy) {
      el.thinking.hidden = false;
      el.thinking.removeAttribute("hidden");
      if (label) el.thinkingText.textContent = label;
    } else {
      el.thinking.hidden = true;
      el.thinking.setAttribute("hidden", "");
      el.thinkingText.textContent = "Working…";
    }
  }

  function setEmpty(empty) {
    el.app.dataset.empty = empty ? "true" : "false";
  }

  /* ---------- harness chrome ---------- */
  function applyHarness(harness) {
    if (!harness) return;
    state.harness = harness;
    const ov = harness.overview || {};
    const status = harness.status || {};
    const running = !!status.running;
    el.app.dataset.running = running ? "true" : "false";

    const runStatus = ov.status || (running ? "active" : "idle");
    el.liveDot.dataset.state = runStatus;
    el.statusLabel.textContent = String(runStatus).replace(/_/g, " ");

    if (ov.run_id) {
      state.focusRunId = ov.run_id;
      el.focusChip.hidden = false;
      el.focusChip.textContent = ov.run_id;
    } else {
      state.focusRunId = null;
      el.focusChip.hidden = true;
    }

    const agentBits = running ? [ov.active_agent, ov.active_task].filter(Boolean) : [];
    el.agentNowText.textContent = agentBits.length
      ? agentBits.join(" · ").slice(0, 100)
      : running
        ? "Harness working…"
        : "Standing by";

    renderStages(ov.pipeline || [], ov.phase);
    renderActiveStrip(harness.active_runs || []);
  }

  function renderStages(pipeline, phase) {
    let steps = Array.isArray(pipeline) ? pipeline.filter((s) => s && (s.phase || s.name)) : [];
    if (!steps.length) {
      const ph = phase && phase !== "idle" ? phase : null;
      let hit = !ph;
      steps = DEFAULT_STAGES.map((p) => {
        if (ph && p === ph) {
          hit = true;
          return { phase: p, state: "active" };
        }
        return { phase: p, state: hit ? "pending" : "done" };
      });
      if (!ph) steps = DEFAULT_STAGES.map((p) => ({ phase: p, state: "pending" }));
    }
    el.stages.innerHTML = steps
      .map((s) => {
        const name = s.phase || s.name || "";
        const st = s.state || "pending";
        return `
      <li data-state="${escapeHtml(st)}" title="${escapeHtml(name)}">
        <span class="node"></span>
        <span class="label">${escapeHtml(phaseLabel(name))}</span>
      </li>`;
      })
      .join("");
    const completed = steps.filter((step) => ["done", "completed"].includes(step.state)).length;
    const active = steps.some((step) => step.state === "active");
    el.stageCount.textContent = `${completed} / ${steps.length}`;
    el.stageRail.dataset.active = active ? "true" : "false";
  }

  function renderActiveStrip(runs) {
    const list = (runs || [])
      .filter((r) => r && r.run_id)
      .slice(0, 6);
    if (!list.length) {
      el.activeStrip.hidden = true;
      el.activeStrip.setAttribute("hidden", "");
      el.activeStrip.innerHTML = "";
      return;
    }
    el.activeStrip.hidden = false;
    el.activeStrip.removeAttribute("hidden");
    el.activeStrip.innerHTML = list
      .map((r) => {
        const label = r.phase || r.status || "live";
        return (
          `<button type="button" class="chip" data-run="${escapeHtml(r.run_id)}">` +
          `${escapeHtml(r.run_id)} · ${escapeHtml(String(label).replace(/_/g, " "))}` +
          `</button>`
        );
      })
      .join("");
  }

  /* ---------- message rendering ---------- */
  function appendMessage(msg, { animate = true } = {}) {
    setEmpty(false);
    const row = document.createElement("article");
    row.className = `msg ${msg.role === "user" ? "user" : "assistant"}`;
    row.dataset.id = msg.id || "";

    const meta = document.createElement("div");
    meta.className = "meta";
    if (msg.role === "assistant" && msg.skill) {
      meta.innerHTML = `<span class="skill-tag">${escapeHtml(msg.skill)}</span>`;
    }
    if (msg.thinking) {
      meta.innerHTML += `<span>${escapeHtml(msg.thinking)}</span>`;
    }
    row.appendChild(meta);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (msg.role === "user") {
      bubble.textContent = msg.content || "";
    } else {
      bubble.innerHTML = `<div class="md">${simpleMarkdown(msg.content || "")}</div>`;
    }
    row.appendChild(bubble);

    if (msg.attachments && msg.attachments.length) {
      const files = document.createElement("div");
      files.className = "message-attachments";
      msg.attachments.forEach((file) => {
        const item = document.createElement("span");
        item.textContent = `📎 ${file.name || "Attached file"}`;
        files.appendChild(item);
      });
      row.appendChild(files);
    }

    if (msg.cards && msg.cards.length) {
      const stack = document.createElement("div");
      stack.className = "card-stack";
      msg.cards.forEach((card) => stack.appendChild(renderCard(card)));
      row.appendChild(stack);
    }

    if (msg.artifacts && msg.artifacts.length) {
      msg.artifacts.forEach((art) => row.appendChild(renderArtifact(art)));
    }

    if (msg.actions && msg.actions.length) {
      const acts = document.createElement("div");
      acts.className = "msg-actions";
      msg.actions.forEach((a) => {
        if (a.href) {
          const link = document.createElement("a");
          link.href = a.href;
          link.textContent = a.label || "Open";
          acts.appendChild(link);
        }
      });
      row.appendChild(acts);
    }

    el.messages.appendChild(row);
    requestAnimationFrame(() => {
      row.classList.add("visible");
      if (animate && window.gsap) {
        gsap.fromTo(row, { y: 10, opacity: 0 }, { y: 0, opacity: 1, duration: 0.35, ease: "power2.out" });
      }
    });
    el.chatMain.scrollTop = el.chatMain.scrollHeight;
    return row;
  }

  function renderCard(card) {
    const kind = card.kind || "generic";
    const data = card.data || {};
    const wrap = document.createElement("div");
    wrap.className = `sg-card ${kind.replace(/_/g, "-")}-card`;

    if (kind === "run_overview") {
      wrap.classList.add("run-overview");
      wrap.innerHTML = `
        <div class="card-title">
          <h4>Run overview</h4>
          <span class="pill ${escapeHtml(data.status || "")}">${escapeHtml(data.status || "idle")}</span>
        </div>
        <dl class="grid-meta">
          <div><dt>Run</dt><dd><code>${escapeHtml(data.run_id || "—")}</code></dd></div>
          <div><dt>Phase</dt><dd>${escapeHtml((data.phase || "idle").replace(/_/g, " "))}</dd></div>
          <div><dt>Topic</dt><dd>${escapeHtml(data.topic || data.goal || "—")}</dd></div>
          <div><dt>Now</dt><dd>${escapeHtml([data.active_agent, data.active_task].filter(Boolean).join(" · ") || "—")}</dd></div>
        </dl>
        <details>
          <summary>Plan &amp; experiments</summary>
          <ul class="plan-list">
            ${(data.plan_experiments || [])
              .map(
                (e) =>
                  `<li><strong>${escapeHtml(e.title || "Experiment")}</strong> — ${escapeHtml(e.status || "planned")}` +
                  (e.hypothesis ? `<br><span>${escapeHtml(e.hypothesis)}</span>` : "") +
                  `</li>`
              )
              .join("") || "<li><em>None recorded yet</em></li>"}
          </ul>
          <ul class="plan-list">
            ${(data.pipeline || [])
              .map(
                (p) =>
                  `<li>${escapeHtml(p.state === "done" ? "✓" : p.state === "active" ? "●" : "○")} ` +
                  `<code>${escapeHtml(p.phase)}</code></li>`
              )
              .join("")}
          </ul>
        </details>`;
      return wrap;
    }

    if (kind === "stage_rail") {
      wrap.innerHTML = `
        <div class="card-title"><h4>Stage path</h4><span class="pill">${escapeHtml(data.phase || "")}</span></div>
        <ol class="stages" style="overflow:visible">
          ${(data.pipeline || [])
            .map(
              (s) =>
                `<li data-state="${escapeHtml(s.state || "pending")}"><span class="node"></span>` +
                `<span class="label">${escapeHtml(shortPhase(s.phase))}</span></li>`
            )
            .join("")}
        </ol>`;
      return wrap;
    }

    if (kind === "approval") {
      wrap.classList.add("approval-card");
      wrap.innerHTML = `
        <div class="card-title">
          <h4>Approval needed</h4>
          <span class="pill awaiting_approval">awaiting approval</span>
        </div>
        <p style="margin:0;color:var(--ink-soft);font-size:13px">${escapeHtml(data.reason || "Release gate")}</p>
        ${(data.missing || []).length
          ? `<p class="note" style="margin:8px 0 0;font-size:12px;color:var(--muted)">Missing: ${escapeHtml(
              (data.missing || []).join(", ")
            )}</p>`
          : ""}
        <div class="approval-actions">
          <button type="button" class="approve" data-approve="${escapeHtml(data.run_id || "")}">Approve release</button>
          <button type="button" class="reject" data-reject="${escapeHtml(data.run_id || "")}">Reject</button>
        </div>`;
      return wrap;
    }

    if (kind === "active_runs") {
      wrap.classList.add("active-runs");
      wrap.innerHTML = `
        <div class="card-title"><h4>Active runs</h4></div>
        <ul>
          ${(data.runs || [])
            .map(
              (r) =>
                `<li><button type="button" class="chip" data-run="${escapeHtml(r.run_id || "")}" style="border:0;background:transparent;padding:0;font:inherit;color:var(--green);font-weight:650">${escapeHtml(
                  r.run_id || "—"
                )}</button>` +
                `<span>${escapeHtml(r.status || "")} · ${escapeHtml(r.phase || "")}</span></li>`
            )
            .join("") || "<li>None</li>"}
        </ul>`;
      return wrap;
    }

    wrap.innerHTML = `<pre style="margin:0;font:12px var(--mono)">${escapeHtml(
      JSON.stringify(data, null, 2)
    )}</pre>`;
    return wrap;
  }

  function renderArtifact(art) {
    const box = document.createElement("div");
    box.className = "artifact";
    const content = art.content || "";
    const name = art.name || "artifact.txt";
    box.innerHTML = `
      <div class="art-head">
        <span>${escapeHtml(name)}${art.format ? " · " + escapeHtml(art.format) : ""}</span>
        <button type="button" class="dl">Download</button>
      </div>
      ${content ? `<pre>${escapeHtml(String(content).slice(0, 4000))}</pre>` : ""}`;
    $(".dl", box)?.addEventListener("click", () => {
      const blob = new Blob([content || ""], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      URL.revokeObjectURL(a.href);
    });
    return box;
  }

  /* ---------- session / messaging ---------- */
  async function ensureSession() {
    if (state.sessionId) {
      try {
        const data = await api(`/api/chat/session/${state.sessionId}`);
        applyHarness(data.harness);
        el.messages.innerHTML = "";
        const msgs = data.messages || [];
        if (msgs.length) {
          setEmpty(false);
          msgs.forEach((m) => appendMessage(m, { animate: false }));
        } else {
          setEmpty(true);
        }
        state.focusRunId = data.session?.focus_run_id || null;
        return;
      } catch {
        state.sessionId = null;
        localStorage.removeItem("sg_chat_session");
      }
    }
    const data = await api("/api/chat/session", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.sessionId = data.session?.id;
    localStorage.setItem("sg_chat_session", state.sessionId);
    applyHarness(data.harness);
    setEmpty(true);
  }

  async function loadChatHistory() {
    try {
      const data = await api("/api/chat/sessions");
      const sessions = data.sessions || [];
      el.historyList.replaceChildren();
      el.historyCount.textContent = String(sessions.length);
      el.historyEmpty.textContent = "Your conversations will appear here.";
      el.historyEmpty.removeAttribute("title");
      el.historyEmpty.hidden = sessions.length > 0;
      sessions.forEach((session) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "history-item";
        item.dataset.session = session.id;
        item.setAttribute("aria-current", session.id === state.sessionId ? "true" : "false");

        const title = document.createElement("span");
        title.className = "history-item-title";
        title.textContent = session.title || "New chat";
        const meta = document.createElement("span");
        meta.className = "history-item-meta";
        const updated = new Date(session.updated_at * 1000);
        meta.textContent = Number.isNaN(updated.getTime())
          ? `${session.message_count} messages`
          : `${updated.toLocaleDateString()} · ${session.message_count} messages`;
        item.append(title, meta);
        item.addEventListener("click", () => openChatSession(session.id));
        el.historyList.appendChild(item);
      });
    } catch (err) {
      el.historyList.replaceChildren();
      el.historyCount.textContent = "—";
      el.historyEmpty.textContent = err.message.includes("Not Found")
        ? "Restart the chat server to load your conversation history."
        : `Couldn't load conversations: ${err.message}`;
      el.historyEmpty.title = err.message;
      el.historyEmpty.hidden = false;
    }
  }

  async function openChatSession(sessionId) {
    if (state.busy || sessionId === state.sessionId) return;
    if (state.pendingFiles.length && !window.confirm("Discard the attached files and open this conversation?")) return;
    try {
      const data = await api(`/api/chat/session/${sessionId}`);
      state.sessionId = sessionId;
      state.focusRunId = data.session?.focus_run_id || null;
      state.pendingFiles = [];
      localStorage.setItem("sg_chat_session", sessionId);
      el.composer.value = "";
      renderPendingFiles();
      autosize();
      el.messages.replaceChildren();
      (data.messages || []).forEach((message) => appendMessage(message, { animate: false }));
      setEmpty(!(data.messages || []).length);
      applyHarness(data.harness);
      el.chatMain.scrollTop = el.chatMain.scrollHeight;
      await loadChatHistory();
    } catch (err) {
      appendMessage({
        role: "assistant",
        content: `Could not open that conversation: ${err.message}`,
        skill: "chat",
      });
    }
  }

  async function sendText(text) {
    const content = (text || "").trim();
    if (!content || state.busy) return;
    await sendMessage(content);
  }

  async function sendMessage(content, files = []) {
    if ((!content && !files.length) || state.busy) return;
    const message = content || `Please review ${files.length === 1 ? "this attached file" : "these attached files"}.`;
    appendMessage({
      role: "user",
      content: message,
      attachments: files.map((file) => ({ name: file.name })),
    });
    el.composer.value = "";
    state.pendingFiles = [];
    renderPendingFiles();
    autosize();
    setBusy(true, files.length ? "Reading your files and prompt…" : "Routing through harness skills…");
    try {
      let body;
      if (files.length) {
        body = new FormData();
        body.append("message", message);
        body.append("session_id", state.sessionId || "");
        if (state.focusRunId) body.append("focus_run_id", state.focusRunId);
        files.forEach((file) => body.append("files", file, file.name));
      } else {
        body = JSON.stringify({
          message,
          session_id: state.sessionId,
          focus_run_id: state.focusRunId,
        });
      }
      const data = await api(files.length ? "/api/chat/upload" : "/api/chat/message", {
        method: "POST",
        body,
      });
      if (data.session_id) {
        state.sessionId = data.session_id;
        localStorage.setItem("sg_chat_session", state.sessionId);
      }
      if (data.message) appendMessage(data.message);
      applyHarness(data.harness);
      await loadChatHistory();
    } catch (err) {
      if (files.length) {
        el.composer.value = content;
        state.pendingFiles = files;
        renderPendingFiles();
        autosize();
      }
      appendMessage({
        role: "assistant",
        content: `Something went wrong: ${err.message}`,
        skill: "chat",
      });
    } finally {
      setBusy(false);
      pollHarness();
    }
  }

  async function sendApproval(runId, approve) {
    if (state.busy) return;
    setBusy(true, approve ? "Approving release…" : "Recording rejection…");
    try {
      const data = await api("/api/chat/message", {
        method: "POST",
        body: JSON.stringify({
          session_id: state.sessionId,
          run_id: runId,
          approve: approve ? true : undefined,
          reject: approve ? undefined : true,
        }),
      });
      if (data.message) appendMessage(data.message);
      applyHarness(data.harness);
    } catch (err) {
      appendMessage({ role: "assistant", content: `Approval failed: ${err.message}`, skill: "approve" });
    } finally {
      setBusy(false);
    }
  }

  function renderPendingFiles() {
    el.attachmentList.replaceChildren();
    state.pendingFiles.forEach((file, index) => {
      const chip = document.createElement("div");
      chip.className = "attachment-chip";
      const label = document.createElement("span");
      label.textContent = `${file.name} · ${Math.max(1, Math.round(file.size / 1024))} KB`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${file.name}`);
      remove.addEventListener("click", () => {
        state.pendingFiles.splice(index, 1);
        renderPendingFiles();
      });
      chip.append(label, remove);
      el.attachmentList.appendChild(chip);
    });
  }

  function queueFiles(files) {
    if (!files?.length || state.busy) return;
    for (const file of files) {
      const exists = state.pendingFiles.some(
        (queued) => queued.name === file.name && queued.size === file.size && queued.lastModified === file.lastModified
      );
      if (!exists) state.pendingFiles.push(file);
    }
    renderPendingFiles();
  }

  async function pollHarness() {
    try {
      const q = state.focusRunId ? `?run_id=${encodeURIComponent(state.focusRunId)}` : "";
      const data = await api(`/api/chat/harness${q}`);
      applyHarness(data);
    } catch {
      /* ignore poll errors */
    }
  }

  function autosize() {
    const t = el.composer;
    t.style.height = "auto";
    t.style.height = Math.min(160, Math.max(44, t.scrollHeight)) + "px";
  }

  /* ---------- events ---------- */
  el.form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(el.composer.value.trim(), state.pendingFiles.slice());
  });

  el.composer.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(el.composer.value.trim(), state.pendingFiles.slice());
    }
  });
  el.composer.addEventListener("input", autosize);

  el.attachBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => {
    queueFiles(el.fileInput.files);
    el.fileInput.value = "";
  });

  // Drag & drop
  ["dragenter", "dragover"].forEach((ev) => {
    window.addEventListener(ev, (e) => {
      e.preventDefault();
      el.form.style.borderColor = "rgba(26,107,74,0.55)";
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    window.addEventListener(ev, (e) => {
      e.preventDefault();
      el.form.style.borderColor = "";
      if (ev === "drop" && e.dataTransfer?.files?.length) {
        queueFiles(e.dataTransfer.files);
      }
    });
  });

  el.newChatBtn.addEventListener("click", async () => {
    localStorage.removeItem("sg_chat_session");
    state.sessionId = null;
    state.focusRunId = null;
    state.pendingFiles = [];
    el.composer.value = "";
    renderPendingFiles();
    autosize();
    el.messages.innerHTML = "";
    setEmpty(true);
    setBusy(false);
    await ensureSession();
    await loadChatHistory();
  });

  el.themeBtn?.addEventListener("click", toggleTheme);

  el.focusChip.addEventListener("click", () => {
    if (state.focusRunId) sendText(`open run ${state.focusRunId}`);
  });

  document.addEventListener("click", (e) => {
    const approve = e.target.closest("[data-approve]");
    if (approve) {
      sendApproval(approve.getAttribute("data-approve"), true);
      return;
    }
    const reject = e.target.closest("[data-reject]");
    if (reject) {
      sendApproval(reject.getAttribute("data-reject"), false);
      return;
    }
    const runBtn = e.target.closest("[data-run]");
    if (runBtn) {
      const rid = runBtn.getAttribute("data-run");
      if (rid) sendText(`open run ${rid}`);
    }
    const prompt = e.target.closest("[data-prompt]");
    if (prompt) sendText(prompt.getAttribute("data-prompt"));
  });

  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "d") {
      e.preventDefault();
      location.href = "/dashboard";
    }
    if (e.key.toLowerCase() === "d" && !e.metaKey && !e.ctrlKey && !e.altKey) {
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "TEXTAREA" || tag === "INPUT" || e.target?.isContentEditable) return;
      toggleTheme();
    }
  });

  /* ---------- boot ---------- */
  (async () => {
    initTheme();
    setBusy(false);
    renderStages([], "idle");
    try {
      await ensureSession();
      await loadChatHistory();
    } catch (err) {
      setEmpty(false);
      appendMessage({
        role: "assistant",
        content: `Could not start chat session: ${err.message}. Is the API up?`,
        skill: "chat",
      });
    }
    setBusy(false);
    pollHarness();
    state.pollTimer = setInterval(pollHarness, 4000);
  })();
})();
