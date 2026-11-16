/* Browser app: progress, spaced repetition, quiz, revision tracking. Runs from file://, no network.
   All progress lives in localStorage under one key; export/import moves it between browsers. */
(function () {
  "use strict";
  const D = window.AIPREP;
  if (!D) return;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const KEY = "aiprep.v1";
  const GRADES = ["0 · blank", "1 · wrong", "2 · partial", "3 · with effort", "4 · solid", "5 · could teach it"];
  const GRADE_HELP = "0 no idea · 1 wrong · 2 partial, big gaps · 3 correct with effort · 4 correct, could explain it to an interviewer · 5 instant, could teach it";

  /* ---------- storage ---------- */
  let canStore = true;
  try { localStorage.setItem("aiprep.test", "1"); localStorage.removeItem("aiprep.test"); } catch (e) { canStore = false; }
  let state = (() => {
    try { const s = JSON.parse(localStorage.getItem(KEY)); if (s && s.v === 1) return s; } catch (e) { /* ignore */ }
    return { v: 1, reviews: {}, q: {}, revised: {}, ex: {}, mistakes: [] };
  })();
  const save = () => { if (canStore) { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { canStore = false; } } };
  if (!canStore) { const b = $("#storage-warning"); if (b) b.hidden = false; }

  /* ---------- dates ---------- */
  const ymd = (d) => d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  const parse = (s) => { const p = s.split("-"); return new Date(+p[0], +p[1] - 1, +p[2]); };
  const today = () => ymd(new Date());
  const addDays = (s, n) => { const d = parse(s); d.setDate(d.getDate() + n); return ymd(d); };
  const daysBetween = (a, b) => Math.round((parse(b) - parse(a)) / 86400000);

  /* ---------- data lookups ---------- */
  const modById = {}, topicById = {}, topicMod = {}, qByTopic = {}, qById = {};
  D.modules.forEach((m) => m.topics.forEach((t) => { modById[m.id] = m; topicById[t.id] = t; topicMod[t.id] = m; }));
  D.questions.forEach((q) => { (qByTopic[q.t] = qByTopic[q.t] || []).push(q); qById[q.id] = q; });
  const sectionFor = (topicId) => {
    const m = topicMod[topicId]; if (!m || !m.page) return null;
    const secs = D.sections[m.id] || [];
    const s = secs.find((x) => x.topics.indexOf(topicId) > -1);
    return s ? m.page + "#" + s.id : m.page;
  };

  /* ---------- spaced repetition (same algorithm as the terminal tracker) ---------- */
  function replay(reviews) {
    let reps = 0, interval = 0, ease = 2.5, last = null, due = null;
    reviews.slice().sort((a, b) => (a.d < b.d ? -1 : a.d > b.d ? 1 : 0)).forEach((r) => {
      const same = last === r.d;
      if (r.g < 3) { reps = 0; interval = 1; }
      else if (!same) { reps += 1; interval = reps === 1 ? 1 : reps === 2 ? 3 : Math.max(1, Math.round(interval * ease)); }
      ease = Math.max(1.3, ease + 0.1 - (5 - r.g) * (0.08 + (5 - r.g) * 0.02));
      last = r.d; due = addDays(r.d, interval);
    });
    return { due, interval, ease };
  }
  function stats(topicId) {
    const rev = state.reviews[topicId] || [];
    if (!rev.length) return { mastery: null, bucket: "unknown", due: null, overdue: 0, n: 0, last: null };
    const sorted = rev.slice().sort((a, b) => (a.d < b.d ? -1 : 1));
    const recent = sorted.slice(-3).map((r) => r.g);
    let m = recent.reduce((a, b) => a + b, 0) / recent.length / 5;
    const sr = replay(rev), t = today();
    const overdue = sr.due && sr.due < t ? daysBetween(sr.due, t) : 0;
    if (overdue) m *= Math.max(0.6, 1 - 0.015 * overdue);
    const mastery = Math.round(m * 100);
    return { mastery, bucket: mastery >= 80 ? "strong" : mastery >= 60 ? "ok" : "weak", due: sr.due, overdue, n: rev.length, last: sorted[sorted.length - 1] };
  }
  function addReview(topicId, g, source, note) {
    const list = (state.reviews[topicId] = state.reviews[topicId] || []);
    const e = { d: today(), g: g, s: source };
    if (note) e.x = note;
    if (source === "rate") { for (let i = list.length - 1; i >= 0; i--) if (list[i].d === e.d && list[i].s === "rate") list.splice(i, 1); }
    list.push(e);
  }
  const bar = (v) => {
    if (v === null || v === undefined) return '<span class="mbar unknown"><i style="width:0"></i><em>?</em></span>';
    const c = v >= 80 ? "strong" : v >= 60 ? "ok" : "weak";
    return `<span class="mbar ${c}"><i style="width:${v}%"></i><em>${v}</em></span>`;
  };
  const stars = (n) => "★".repeat(n) + "☆".repeat(3 - n);
  const allTopicIds = () => D.modules.reduce((a, m) => a.concat(m.topics.map((t) => t.id)), []);
  function moduleMastery(m) {
    const k = m.topics.map((t) => stats(t.id).mastery).filter((x) => x !== null);
    return { mastery: k.length ? Math.round(k.reduce((a, b) => a + b, 0) / k.length) : null, rated: k.length, total: m.topics.length };
  }
  function streak() {
    const days = new Set();
    Object.keys(state.reviews).forEach((t) => state.reviews[t].forEach((r) => days.add(r.d)));
    let d = today(), n = 0;
    if (!days.has(d)) d = addDays(d, -1);
    while (days.has(d)) { n++; d = addDays(d, -1); }
    return n;
  }
  function weekAnswered() {
    const from = addDays(today(), -6); let n = 0;
    Object.keys(state.reviews).forEach((t) => state.reviews[t].forEach((r) => { if (r.s === "quiz" && r.d >= from) n++; }));
    return n;
  }
  const topicLabel = (id) => (topicById[id] ? topicById[id].title : id);
  const modLabel = (id) => (modById[id] ? modById[id].title : id);
  const quizLink = (p) => "quiz.html#" + Object.keys(p).map((k) => k + "=" + encodeURIComponent(p[k])).join("&");

  /* ================= TODAY ================= */
  function pageToday(app) {
    const ids = allTopicIds();
    const st = ids.map((id) => ({ id, s: stats(id) }));
    const rated = st.filter((x) => x.s.mastery !== null);
    const t = today();
    const due = st.filter((x) => x.s.due && x.s.due <= t).sort((a, b) => b.s.overdue - a.s.overdue);
    const weak = st.filter((x) => x.s.bucket === "weak").sort((a, b) => a.s.mastery - b.s.mastery);
    const unknown = st.filter((x) => x.s.bucket === "unknown");
    let h = '<h1 class="page-h">Today</h1>';
    if (!rated.length) {
      h += `<div class="hero-card"><h2>Start here</h2><p>Tell the app what you already know. It takes about 10 minutes, there are no right answers, and it lets the app show you what to work on.</p>
        <a class="btn" href="topics.html">Rate what I know</a> <a class="btn ghost" href="revise.html">Or start reading the notes</a></div>`;
    } else {
      h += `<div class="stats">
        <div class="stat"><b>${due.length}</b><span>topics due for review</span></div>
        <div class="stat"><b>${weak.length}</b><span>weak topics</span></div>
        <div class="stat"><b>${weekAnswered()}</b><span>questions this week</span></div>
        <div class="stat"><b>${streak()}</b><span>day streak</span></div></div>
        <div class="cta-row"><a class="btn big" href="${quizLink({ mode: "mixed", n: 10, go: 1 })}">Start today's 10 questions</a>
        <span class="muted">Due topics first, then your weakest, then new ones. About 15 minutes.</span></div>`;
    }
    h += '<h2 class="sec-h">Your plan for today</h2><ol class="plan">';
    h += `<li><b>Review.</b> Answer questions out loud, then reveal and grade yourself honestly. <a href="${quizLink({ mode: "due", n: 10, go: 1 })}">Due only</a></li>`;
    const w = weak[0] || unknown[0];
    if (w) {
      const sec = sectionFor(w.id);
      h += `<li><b>Read one section you are weak on.</b> ${esc(topicLabel(w.id))} (${esc(modLabel(topicMod[w.id].id))}) ${sec ? `<a href="${sec}">Open the notes</a>` : ""}</li>`;
    } else h += "<li><b>Read one section.</b> Pick any module on the <a href=\"revise.html\">Revise</a> page.</li>";
    h += '<li><b>Solve one exercise.</b> Try one in <a href="exercises.html">Exercises</a>. Write it in your editor first, then compare.</li>';
    h += "<li><b>Note what you got wrong</b> in the box that appears when you grade a question 0-2. Those come back sooner.</li></ol>";
    if (weak.length) {
      h += '<h2 class="sec-h">Weakest topics</h2><div class="rows">';
      weak.slice(0, 6).forEach((x) => {
        const sec = sectionFor(x.id), m = topicMod[x.id];
        h += `<div class="row"><div class="row-main"><b>${esc(topicLabel(x.id))}</b><small>${esc(m.title)}</small></div>${bar(x.s.mastery)}
          <div class="row-actions">${sec ? `<a class="btn small ghost" href="${sec}">Notes</a>` : ""}<a class="btn small" href="${quizLink({ m: m.id, t: topicById[x.id].slug, n: 5, go: 1 })}">Quiz me</a></div></div>`;
      });
      h += "</div>";
    }
    if (due.length) {
      h += `<h2 class="sec-h">Due for review (${due.length})</h2><div class="rows">`;
      due.slice(0, 6).forEach((x) => {
        const m = topicMod[x.id];
        h += `<div class="row"><div class="row-main"><b>${esc(topicLabel(x.id))}</b><small>${esc(m.title)} · ${x.s.overdue ? x.s.overdue + " day(s) overdue" : "due today"}</small></div>${bar(x.s.mastery)}
          <div class="row-actions"><a class="btn small" href="${quizLink({ m: m.id, t: topicById[x.id].slug, n: 5, go: 1 })}">Quiz me</a></div></div>`;
      });
      h += "</div>";
    }
    h += '<h2 class="sec-h">Modules</h2><div class="modgrid">';
    D.modules.forEach((m) => {
      const mm = moduleMastery(m);
      h += `<a class="modcard" href="${m.page || "revise.html"}"><b>${esc(m.title)}</b>${bar(mm.mastery)}<small>${mm.rated} of ${mm.total} topics rated</small></a>`;
    });
    h += "</div>";
    app.insertAdjacentHTML("afterbegin", h);
  }

  /* ================= REVISE (module list) ================= */
  function pageRevise(app) {
    let h = `<h1 class="page-h">Revise</h1><p class="lede">One note per module. Read a section, tick it as revised, then test yourself on it. Notes explain the idea, show runnable examples, and flag the traps interviewers probe.</p><div class="modgrid wide">`;
    D.modules.forEach((m) => {
      const mm = moduleMastery(m), secs = D.sections[m.id] || [];
      const done = secs.filter((s) => state.revised[m.id + "#" + s.id]).length;
      h += `<div class="modcard big"><b>${esc(m.title)}</b>
        <small>${m.topics.length} topics${secs.length ? " · " + secs.length + " sections" : ""}</small>
        ${secs.length ? `<div class="mini"><span>Revised</span><span class="tbar"><i style="width:${done / secs.length * 100}%"></i></span><em>${done}/${secs.length}</em></div>` : "<small>Notes not written yet</small>"}
        <div class="mini"><span>Mastery</span>${bar(mm.mastery)}</div>
        <div class="row-actions">${m.page ? `<a class="btn small" href="${m.page}">Read notes</a>` : ""}<a class="btn small ghost" href="${quizLink({ m: m.id, n: 10, go: 1 })}">Quiz module</a></div></div>`;
    });
    app.insertAdjacentHTML("afterbegin", h + "</div>");
  }

  /* ================= NOTE PAGE ================= */
  function pageNote() {
    const modId = document.body.dataset.module, m = modById[modId]; if (!m) return;
    const secs = D.sections[modId] || [];
    const keyOf = (id) => modId + "#" + id;
    $$(".prose h2[data-topics]").forEach((h2) => {
      const topics = h2.dataset.topics ? h2.dataset.topics.split(",").filter(Boolean) : [];
      const div = document.createElement("div"); div.className = "secbar";
      const slugs = topics.map((t) => topicById[t] ? topicById[t].slug : "").filter(Boolean).join(",");
      const nq = topics.reduce((a, t) => a + (qByTopic[t] || []).length, 0);
      div.innerHTML = `<label class="revised"><input type="checkbox"> Mark as revised</label>` +
        (nq ? `<a class="btn small ghost" href="${quizLink({ m: modId, t: slugs, n: Math.min(5, nq), go: 1 })}">Quiz me on this (${nq})</a>` : `<a class="btn small ghost" href="${quizLink({ m: modId, n: 5, go: 1 })}">Quiz this module</a>`);
      h2.after(div);
      const cb = $("input", div); cb.checked = !!state.revised[keyOf(h2.id)];
      cb.addEventListener("change", () => { if (cb.checked) state.revised[keyOf(h2.id)] = today(); else delete state.revised[keyOf(h2.id)]; save(); refresh(); });
    });
    function refresh() {
      const done = secs.filter((s) => state.revised[keyOf(s.id)]).length;
      $("#note-progress").innerHTML = `<div class="np"><b>${done} of ${secs.length} sections revised</b><span class="tbar"><i style="width:${secs.length ? done / secs.length * 100 : 0}%"></i></span>
        <a class="btn small" href="${quizLink({ m: modId, n: 10, go: 1 })}">Quiz this module</a></div>`;
      $$(".toc a").forEach((a) => a.classList.toggle("done", !!state.revised[keyOf(a.dataset.target)]));
    }
    refresh();
  }

  /* ================= QUIZ ================= */
  function parseHash() {
    const p = {}; (location.hash || "").replace(/^#/, "").split("&").forEach((kv) => { if (!kv) return; const i = kv.indexOf("="); p[decodeURIComponent(kv.slice(0, i))] = decodeURIComponent(kv.slice(i + 1)); });
    return p;
  }
  function pool(o) {
    let qs = D.questions;
    if (o.m) qs = qs.filter((q) => q.m === o.m);
    if (o.t) { const set = o.t.split(",").map((s) => (o.m ? o.m + "." + s : s)); qs = qs.filter((q) => set.indexOf(q.t) > -1); }
    if (o.l) qs = qs.filter((q) => String(q.l) === String(o.l));
    if (o.k) qs = qs.filter((q) => q.k === o.k);
    if (o.mode === "due") { const t = today(); qs = qs.filter((q) => { const s = stats(q.t); return s.due && s.due <= t; }); }
    return qs;
  }
  function choose(o) {
    const qs = pool(o), n = +o.n || 10, t = today();
    if (o.mode === "random") return qs.slice().sort(() => Math.random() - 0.5).slice(0, n);
    const scored = qs.map((q) => {
      const s = stats(q.t), rec = state.q[q.id];
      let sc = (s.mastery === null ? 50 : s.mastery);
      if (rec) { sc += 8 * rec.g[rec.g.length - 1] - Math.min(30, daysBetween(rec.last, t)); } else sc -= 15;
      if (o.mode === "weak" || o.mode === "mixed") { if (s.mastery !== null) sc -= (100 - s.mastery) / 2; }
      if (o.mode === "mixed" && s.due && s.due <= t) sc -= 30;
      return { q, sc: sc + Math.random() * 10 };
    });
    return scored.sort((a, b) => a.sc - b.sc).slice(0, n).map((x) => x.q);
  }
  function pageQuiz(app) {
    let opts = Object.assign({ mode: "mixed", n: 10 }, parseHash());
    let run = null;
    function setup() {
      const mods = ['<option value="">All modules</option>'].concat(D.modules.map((m) => `<option value="${m.id}"${opts.m === m.id ? " selected" : ""}>${esc(m.title)}</option>`)).join("");
      const sel = (name, items) => items.map((i) => `<option value="${i[0]}"${String(opts[name] || "") === i[0] ? " selected" : ""}>${i[1]}</option>`).join("");
      const topicNote = opts.t ? `<p class="chip">Topic filter: ${esc(opts.t.split(",").map((s) => (topicById[opts.m + "." + s] || { title: s }).title).join(", "))} <a href="#" data-clear-t>clear</a></p>` : "";
      app.innerHTML = `<h1 class="page-h">Quiz</h1><p class="lede">Read the question and answer out loud, as if an interviewer asked it. Reveal the key points, then grade yourself honestly. Low grades bring the topic back sooner.</p>${topicNote}
        <div class="form-grid">
        <label>Module<select data-o="m">${mods}</select></label>
        <label>Questions<select data-o="n">${sel("n", [["5", "5"], ["10", "10"], ["15", "15"], ["20", "20"]])}</select></label>
        <label>Pick<select data-o="mode">${sel("mode", [["mixed", "What I should work on"], ["due", "Only topics due"], ["weak", "Weakest first"], ["random", "Random"]])}</select></label>
        <label>Level<select data-o="l">${sel("l", [["", "Any"], ["1", "★ fundamentals"], ["2", "★★ working engineer"], ["3", "★★★ senior / 3-4 years"]])}</select></label>
        <label>Type<select data-o="k">${sel("k", [["", "Any"], ["concept", "Concept"], ["predict", "Predict the output"], ["debug", "Debug"], ["design", "Design"], ["compare", "Compare"]])}</select></label></div>
        <p class="muted" data-pool></p><button type="button" class="btn big" data-start>Start</button>`;
      $$("[data-o]", app).forEach((s) => s.addEventListener("change", () => { opts[s.dataset.o] = s.value; if (s.dataset.o === "m") delete opts.t; count(); }));
      const clr = $("[data-clear-t]", app); if (clr) clr.addEventListener("click", (e) => { e.preventDefault(); delete opts.t; setup(); });
      $("[data-start]", app).addEventListener("click", start);
      count();
    }
    function count() { const n = pool(opts).length; $("[data-pool]", app).textContent = n ? n + " matching questions." : "No matching questions. " + (opts.mode === "due" ? "Nothing is due yet; try \"What I should work on\"." : "Try other filters."); $("[data-start]", app).disabled = !n; }
    function start() { run = { list: choose(opts), i: 0, results: [] }; if (!run.list.length) return; show(); }
    function show() {
      const q = run.list[run.i], t = topicById[q.t], m = modById[q.m];
      const secLink = sectionFor(q.t);
      app.innerHTML = `<div class="qprog"><span>Question ${run.i + 1} of ${run.list.length}</span><span class="tbar"><i style="width:${run.i / run.list.length * 100}%"></i></span><a href="#" data-quit>End</a></div>
        <div class="qcard"><p class="qmeta">${esc(m.title)} · ${esc(t.title)} · <span title="level">${stars(q.l)}</span> · ${esc(q.k)}</p>
        <div class="prose qtext">${q.q}</div>
        <textarea class="ta" rows="3" placeholder="Your answer (optional: say it out loud first)" aria-label="Your answer"></textarea>
        <button type="button" class="btn" data-reveal>Show key points</button>
        <div class="reveal" hidden><div class="prose ans"><strong class="ans-h">Key points</strong>${q.a}</div>
        ${q.f.length ? `<div class="follow"><b>Likely follow-ups</b><ul>${q.f.map((f) => "<li>" + f + "</li>").join("")}</ul></div>` : ""}
        ${secLink ? `<p class="muted"><a href="${secLink}" target="_blank" rel="noopener">Open the notes for this topic</a> (new tab)</p>` : ""}
        <p class="grade-h">How well did you know it?</p><div class="gradebar">${GRADES.map((g, i) => `<button type="button" data-g="${i}" class="gb g${i}">${g}</button>`).join("")}</div>
        <p class="muted small">${GRADE_HELP}</p>
        <label class="mistake">Note what went wrong (optional, one sentence, used if you grade 0-2)<input type="text" placeholder="e.g. forgot that gather does not cancel the others on error"></label></div></div>`;
      $("[data-quit]", app).addEventListener("click", (e) => { e.preventDefault(); summary(); });
      $("[data-reveal]", app).addEventListener("click", reveal);
      function reveal() { $(".reveal", app).hidden = false; $("[data-reveal]", app).hidden = true; $(".reveal", app).scrollIntoView({ block: "nearest" }); }
      $$("[data-g]", app).forEach((b) => b.addEventListener("click", () => grade(+b.dataset.g)));
      function grade(g) {
        const note = g <= 2 ? ($(".mistake input", app).value || "").trim() : "";
        addReview(q.t, g, "quiz", note);
        const rec = (state.q[q.id] = state.q[q.id] || { n: 0, g: [], last: today() });
        rec.n++; rec.last = today(); rec.g = rec.g.concat(g).slice(-5);
        if (note) state.mistakes.push({ d: today(), t: q.t, x: note });
        save(); run.results.push({ q, g });
        run.i++; if (run.i >= run.list.length) summary(); else show();
      }
      const reveal2 = () => { if ($(".reveal", app).hidden) reveal(); };
      document.onkeydown = (e) => {
        const tag = (e.target.tagName || "").toLowerCase();
        if (tag === "textarea" || tag === "input") { if (e.key === "Enter" && e.ctrlKey) reveal2(); return; }
        if (e.key === " " || e.key === "Enter") { e.preventDefault(); reveal2(); }
        else if (/^[0-5]$/.test(e.key) && !$(".reveal", app).hidden) grade(+e.key);
      };
    }
    function summary() {
      document.onkeydown = null;
      const rs = run.results;
      if (!rs.length) { setup(); return; }
      const avg = rs.reduce((a, r) => a + r.g, 0) / rs.length;
      const low = rs.filter((r) => r.g <= 2);
      let h = `<h1 class="page-h">Done: ${avg.toFixed(1)} out of 5</h1><p class="lede">${rs.length} question${rs.length === 1 ? "" : "s"} answered.</p>`;
      if (low.length) {
        h += '<h2 class="sec-h">Revise these</h2><div class="rows">';
        const seen = new Set();
        low.forEach((r) => {
          if (seen.has(r.q.t)) return; seen.add(r.q.t);
          const sec = sectionFor(r.q.t);
          h += `<div class="row"><div class="row-main"><b>${esc(topicLabel(r.q.t))}</b><small>${esc(modLabel(r.q.m))}</small></div><span class="g g${r.g}">${r.g}</span>
            <div class="row-actions">${sec ? `<a class="btn small" href="${sec}">Read the notes</a>` : ""}</div></div>`;
        });
        h += "</div>";
      } else h += '<p class="panel">Nothing below 3. Solid session.</p>';
      h += `<div class="cta-row"><button type="button" class="btn" data-again>Another round</button><a class="btn ghost" href="index.html">Back to Today</a></div>`;
      app.innerHTML = h;
      $("[data-again]", app).addEventListener("click", () => { run = null; setup(); });
    }
    function boot() {
      opts = Object.assign({ mode: "mixed", n: 10 }, parseHash()); run = null; document.onkeydown = null;
      setup();
      if (opts.go === "1" && pool(opts).length) { delete opts.go; start(); }
    }
    window.addEventListener("hashchange", boot);
    boot();
  }

  /* ================= MY TOPICS ================= */
  function pageTopics(app) {
    let filter = "all", text = "";
    function render() {
      const open = new Set($$("details.tm[open]", app).map((d) => d.dataset.m));
      let h = `<h1 class="page-h">My topics</h1><p class="lede">Rate how well you know each topic (0 = no idea, 5 = could teach it). Be honest: the app uses this to decide what to show you. Quizzes and exercises update it too.</p>
        <div class="toolbar"><select data-f aria-label="Filter">${[["all", "All topics"], ["unrated", "Not rated yet"], ["weak", "Weak"], ["due", "Due for review"]].map((f) => `<option value="${f[0]}"${filter === f[0] ? " selected" : ""}>${f[1]}</option>`).join("")}</select>
        <input type="search" class="search" placeholder="Search topics" value="${esc(text)}" data-s aria-label="Search topics"></div>`;
      D.modules.forEach((m) => {
        const mm = moduleMastery(m), t = today();
        const rows = m.topics.filter((tp) => {
          const s = stats(tp.id);
          if (text && (tp.title + " " + tp.slug).toLowerCase().indexOf(text.toLowerCase()) < 0) return false;
          if (filter === "unrated") return s.mastery === null;
          if (filter === "weak") return s.bucket === "weak";
          if (filter === "due") return s.due && s.due <= t;
          return true;
        });
        if (!rows.length) return;
        h += `<details class="tm" data-m="${m.id}"${open.has(m.id) || filter !== "all" || text ? " open" : ""}><summary><b>${esc(m.title)}</b>${bar(mm.mastery)}<small>${mm.rated}/${mm.total} rated</small></summary><div class="rows">`;
        rows.forEach((tp) => {
          const s = stats(tp.id), cur = s.last ? s.last.g : -1;
          h += `<div class="row trow" data-t="${tp.id}"><div class="row-main"><b>${esc(tp.title)}</b><small>${s.due ? (s.due <= t ? "due now" : "next review " + s.due) : "not rated"}</small></div>${bar(s.mastery)}
            <div class="chips" role="group" aria-label="Rate ${esc(tp.title)}">${[0, 1, 2, 3, 4, 5].map((g) => `<button type="button" data-g="${g}" class="chip-b g${g}${cur === g && s.last.s === "rate" ? " on" : ""}">${g}</button>`).join("")}</div></div>`;
        });
        h += "</div></details>";
      });
      h += `<h2 class="sec-h">Mistakes you logged</h2>${state.mistakes.length ? '<ul class="mist">' + state.mistakes.slice(-30).reverse().map((x) => `<li><b>${esc(topicLabel(x.t))}</b> <small>${x.d}</small><br>${esc(x.x)}</li>`).join("") + "</ul>" : '<p class="muted">Nothing yet. Grade a question 0-2 and add a sentence about what went wrong.</p>'}
        <h2 class="sec-h">Your data</h2><p class="muted">Progress is saved in this browser only. Export a backup to move it to another browser or computer.</p>
        <div class="cta-row"><button type="button" class="btn ghost" data-export>Export progress</button><label class="btn ghost filebtn">Import progress<input type="file" accept="application/json" data-import hidden></label><button type="button" class="btn ghost danger" data-reset>Reset everything</button></div><p class="muted" data-dmsg></p>`;
      app.innerHTML = h;
      wire();
    }
    function wire() {
      $("[data-f]", app).addEventListener("change", (e) => { filter = e.target.value; render(); });
      const s = $("[data-s]", app); s.addEventListener("input", () => { text = s.value; const pos = s.selectionStart; render(); const n = $("[data-s]", app); n.focus(); n.setSelectionRange(pos, pos); });
      $$(".trow [data-g]", app).forEach((b) => b.addEventListener("click", () => { addReview(b.closest(".trow").dataset.t, +b.dataset.g, "rate"); save(); render(); }));
      const msg = $("[data-dmsg]", app);
      $("[data-export]", app).addEventListener("click", () => {
        const blob = new Blob([JSON.stringify(state, null, 1)], { type: "application/json" }), a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = "ai-prep-progress-" + today() + ".json"; document.body.appendChild(a); a.click(); a.remove();
        msg.textContent = "Exported. Keep the file somewhere safe.";
      });
      $("[data-import]", app).addEventListener("change", (e) => {
        const f = e.target.files[0]; if (!f) return; const r = new FileReader();
        r.onload = () => { try { const s2 = JSON.parse(r.result); if (!s2 || s2.v !== 1) throw new Error("not a progress file"); state = s2; save(); render(); } catch (err) { msg.textContent = "That file is not a valid progress export."; } };
        r.readAsText(f);
      });
      const rs = $("[data-reset]", app);
      rs.addEventListener("click", () => { if (rs.dataset.sure) { state = { v: 1, reviews: {}, q: {}, revised: {}, ex: {}, mistakes: [] }; save(); render(); } else { rs.dataset.sure = "1"; rs.textContent = "Click again to erase all progress"; } });
    }
    render();
  }

  /* ================= EXERCISES ================= */
  function pageExercises() {
    const MAP = { 4: "Solved on my own", 3: "Solved with hints", 2: "Needed the solution" };
    function paint() {
      let done = 0, total = 0;
      $$(".exmod").forEach((sec) => {
        let d = 0; const items = $$("details.ex", sec);
        items.forEach((ex) => {
          total++; const rec = state.ex[ex.dataset.ex], st = $("[data-status]", ex);
          if (rec) { d += rec.r >= 3 ? 1 : 0; st.textContent = MAP[rec.r]; st.className = "ex-status r" + rec.r; } else { st.textContent = ""; st.className = "ex-status"; }
          $$("[data-r]", ex).forEach((b) => b.classList.toggle("on", !!rec && String(rec.r) === b.dataset.r));
        });
        done += d; $("[data-count]", sec).textContent = " " + d + "/" + items.length;
      });
      $("#ex-summary").textContent = done + " of " + total + " exercises solved without needing the solution.";
    }
    $$("details.ex").forEach((ex) => {
      $$("[data-r]", ex).forEach((b) => b.addEventListener("click", () => {
        const id = ex.dataset.ex, topics = ex.dataset.topics ? ex.dataset.topics.split(",") : [];
        topics.forEach((t) => { const list = state.reviews[t] || []; for (let i = list.length - 1; i >= 0; i--) if (list[i].s === "ex:" + id) list.splice(i, 1); });
        if (b.dataset.r === "clear") delete state.ex[id];
        else { const r = +b.dataset.r; state.ex[id] = { d: today(), r: r }; topics.forEach((t) => addReview(t, r, "ex:" + id)); }
        save(); paint();
      }));
    });
    paint();
    if (location.hash) { const el = document.getElementById(location.hash.slice(1)); if (el) el.scrollIntoView(); }
  }

  /* ---------- boot ---------- */
  const page = document.body.dataset.page, app = $("#app");
  if (page === "today" && app) pageToday(app);
  else if (page === "revise" && app) pageRevise(app);
  else if (page === "quiz" && app) pageQuiz(app);
  else if (page === "topics" && app) pageTopics(app);
  else if (page === "note") pageNote();
  else if (page === "exercises") pageExercises();
})();
