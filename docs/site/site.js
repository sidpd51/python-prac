/* Offline notes: outline highlight, copy buttons, and small interactive widgets. Vanilla JS, no network. */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); };
  function store(key, val) {
    try { if (val === undefined) return JSON.parse(localStorage.getItem(key) || "null"); localStorage.setItem(key, JSON.stringify(val)); } catch (e) { return null; }
  }
  function copy(text, btn) {
    function done() { if (!btn) return; var t = btn.textContent; btn.textContent = "Copied"; setTimeout(function () { btn.textContent = t; }, 1200); }
    try { navigator.clipboard.writeText(text).then(done, fallback); } catch (e) { fallback(); }
    function fallback() {
      var ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e) {} ta.remove();
    }
  }

  /* ---------- outline highlight ---------- */
  (function () {
    var links = $$(".toc a"); if (!links.length) return;
    var map = {}; links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    function mark(on) { links.forEach(function (a) { a.classList.toggle("on", a === on); }); }
    links.forEach(function (a) { a.addEventListener("click", function () { setTimeout(function () { mark(a); }, 0); }); });
    if (!("IntersectionObserver" in window)) return;
    var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting && map[e.target.id]) mark(map[e.target.id]); }); }, { rootMargin: "-90px 0px -70% 0px" });
    $$(".prose h2").forEach(function (h) { io.observe(h); });
  })();

  /* ---------- copy buttons on code blocks ---------- */
  $$(".prose pre, .ex-body pre").forEach(function (pre) {
    if (pre.closest(".out")) return;
    var wrap = pre.closest(".hl") || pre; wrap.classList.add("has-copy");
    var b = document.createElement("button"); b.type = "button"; b.className = "copy"; b.textContent = "Copy"; b.setAttribute("aria-label", "Copy code");
    b.addEventListener("click", function () { copy(pre.innerText.replace(/\n$/, ""), b); });
    wrap.appendChild(b);
  });

  /* ---------- tabs (solution / stub / test) ---------- */
  $$(".tabset").forEach(function (set) {
    var panels = $$(":scope > .tabpanel", set); if (!panels.length) return;
    var bar = document.createElement("div"); bar.className = "tabbar"; bar.setAttribute("role", "tablist");
    set.insertBefore(bar, set.firstChild); set.classList.add("live");
    function pick(i) { panels.forEach(function (p, j) { p.hidden = i !== j; }); $$("button", bar).forEach(function (b, j) { b.setAttribute("aria-selected", i === j ? "true" : "false"); }); }
    panels.forEach(function (p, i) {
      var b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "tab"); b.textContent = p.dataset.label;
      b.addEventListener("click", function () { pick(i); }); bar.appendChild(b);
    });
    pick(0);
  });

  /* ---------- generic "click a node, read a panel" ---------- */
  function clickable(root, sel) {
    var panel = $("[data-panel]", root);
    function show(el) {
      $$(sel, root).forEach(function (n) { n.classList.toggle("sel", n === el); });
      var cmd = el.dataset.cmd ? '<div class="cmdline"><code>' + esc(el.dataset.cmd) + '</code><button type="button" class="copy inline">Copy</button></div>' : "";
      panel.innerHTML = "<strong>" + esc(el.dataset.title) + ".</strong> " + el.dataset.body + cmd;
      var b = $(".copy", panel); if (b) b.addEventListener("click", function () { copy(el.dataset.cmd, b); });
    }
    $$(sel, root).forEach(function (n) {
      n.addEventListener("click", function () { show(n); });
      n.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); show(n); } });
    });
  }
  ["system-map", "arch", "flow"].forEach(function (name) { $$('[data-widget="' + name + '"]').forEach(function (w) { clickable(w, name === "flow" ? "li" : ".node"); }); });

  /* ---------- daily loop ---------- */
  $$('[data-widget="daily-loop"]').forEach(function (w) {
    var steps = [
      { t: "Review", m: 20, href: "quiz.html#mode=mixed&n=10&go=1", label: "Start 10 questions", d: "Answer aloud as if an interviewer were in the room, reveal the key points, then grade yourself 0-5. Low grades come back tomorrow." },
      { t: "Revise", m: 25, href: "revise.html", label: "Open the notes", d: "Read one section of a note you are weak on, tick it as revised, then press 'Quiz me on this'." },
      { t: "Practise", m: 40, href: "exercises.html", label: "Open exercises", d: "Solve one or two exercises. Stuck for 15 minutes: read the problem again. Stuck for 30: open the solution, close it, and redo it from memory tomorrow." },
      { t: "Close the loop", m: 5, href: "topics.html", label: "Open my topics", d: "Note anything you got wrong in the box that appears when you grade 0-2, and re-rate topics that changed." }
    ];
    var bar = $(".loop-bar", w), panel = $("[data-panel]", w), total = steps.reduce(function (a, s) { return a + s.m; }, 0);
    steps.forEach(function (s, i) {
      var b = document.createElement("button"); b.type = "button"; b.className = "loop-seg s" + i; b.style.flexGrow = s.m; b.setAttribute("role", "tab");
      b.innerHTML = "<span>" + s.t + "</span><small>" + s.m + " min</small>";
      b.addEventListener("click", function () {
        $$(".loop-seg", bar).forEach(function (x) { x.setAttribute("aria-selected", x === b ? "true" : "false"); });
        panel.innerHTML = "<strong>" + s.t + " · " + s.m + " of " + total + " min.</strong> " + s.d + '<div class="cmdline"><a class="btn small" href="' + s.href + '">' + esc(s.label) + "</a></div>";
      });
      bar.appendChild(b); if (i === 0) b.click();
    });
  });

  /* ---------- spaced repetition + mastery ---------- */
  $$('[data-widget="srs"]').forEach(function (w) {
    var reviews = [], tbody = $("tbody", w), line = $(".srs-line", w);
    function replay() {
      var reps = 0, interval = 0, ease = 2.5, day = 0, rows = [];
      reviews.forEach(function (g, i) {
        if (g < 3) { reps = 0; interval = 1; }
        else { reps += 1; interval = reps === 1 ? 1 : reps === 2 ? 3 : Math.max(1, Math.round(interval * ease)); }
        ease = Math.max(1.3, ease + 0.1 - (5 - g) * (0.08 + (5 - g) * 0.02));
        rows.push({ n: i + 1, day: day, g: g, next: interval, ease: ease });
        day += interval;
      });
      return rows;
    }
    function draw() {
      var rows = replay();
      tbody.innerHTML = rows.length ? rows.map(function (r) {
        return '<tr><td>' + r.n + '</td><td>day ' + r.day + '</td><td><span class="g g' + r.g + '">' + r.g + '</span></td><td>' + r.next + ' day' + (r.next === 1 ? "" : "s") + '</td><td>' + r.ease.toFixed(2) + '</td></tr>';
      }).join("") : '<tr><td colspan="5" class="muted">Press a grade to log your first review.</td></tr>';
      var span = rows.length ? rows[rows.length - 1].day + rows[rows.length - 1].next : 1, h = "";
      rows.forEach(function (r) { h += '<i class="dot g' + r.g + '" style="left:' + (r.day / span * 100) + '%"></i>'; });
      line.innerHTML = rows.length ? h + '<em>day ' + span + ': next review</em>' : "";
    }
    $$("[data-g]", w).forEach(function (b) { b.addEventListener("click", function () { if (reviews.length < 14) reviews.push(+b.dataset.g); draw(); }); });
    $("[data-reset]", w).addEventListener("click", function () { reviews = []; draw(); });
    draw();
    var v = {};
    function mastery() {
      $$("[data-in]", w).forEach(function (i) { v[i.dataset.in] = +i.value; $('[data-o="' + i.dataset.in + '"]', w).textContent = i.value; });
      var sig = [v.avg / 5], att = v.pass + v.fail; if (att) sig.push(v.pass / att);
      var m = sig.reduce(function (a, b) { return a + b; }, 0) / sig.length;
      if (v.over) m *= Math.max(0.6, 1 - 0.015 * v.over);
      var pct = Math.round(m * 100), bucket = pct >= 80 ? "strong" : pct >= 60 ? "ok" : "weak";
      var fill = $("[data-meter]", w); fill.style.width = pct + "%"; fill.className = "meter-fill " + bucket;
      $("[data-mnum]", w).textContent = pct + " · " + bucket;
      $("[data-mnote]", w).textContent = "Average of your grade (" + (v.avg / 5 * 100).toFixed(0) + "%)" + (att ? " and your test pass rate (" + Math.round(v.pass / att * 100) + "%)" : "") + (v.over ? ", decayed " + Math.round((1 - Math.max(0.6, 1 - 0.015 * v.over)) * 100) + "% for being overdue" : "") + ". 80 or more is strong, below 60 is weak.";
    }
    $$("[data-in]", w).forEach(function (i) { i.addEventListener("input", mastery); }); mastery();
  });

  /* ---------- command finder ---------- */
  $$('[data-widget="commands"]').forEach(function (w) {
    var cmds = [
      ["study today", "Daily plan: due reviews, weakest topics, next milestone", "plan daily start"],
      ["study report [MODULE]", "Strengths and weaknesses, per module or per topic", "weak strong progress"],
      ["study baseline [MODULE]", "Rate every topic 0-5 once, so weak spots show up", "rate first start"],
      ["study quiz --due", "Interview questions for topics due for review", "quiz revise spaced review"],
      ["study quiz -m async -l 3 -k predict", "Filter by module, level (1-3) and kind", "quiz filter level predict debug design"],
      ["study quiz --weak -n 5", "Bias the questions towards your weak topics", "quiz weak"],
      ["study test [MODULE|capstone]", "Run exercises; results update your progress", "test run pytest exercise"],
      ["study exercises [MODULE]", "List exercises with solved, failing or todo status", "exercises status"],
      ["study rate TOPIC 0-5", "Log a self-rating after revising a topic", "rate grade"],
      ["study mistake \"TEXT\" -t TOPIC", "Log a misconception; the topic returns tomorrow", "mistake log wrong"],
      ["study topics [MODULE] -v", "List topic ids", "topics ids list"],
      ["study start MODULE / finish MODULE", "Move a new module in and out of your stats", "new module planned learning covered"],
      ["study scaffold MODULE", "Create exercise, solution and question templates", "new module create template"],
      ["study verify [MODULE|capstone]", "Run the tests against the reference solutions", "author verify solution"],
      ["study check [MODULE]", "Validate content and show topics with no practice", "author check coverage"],
      ["study stubs [MODULE]", "Generate missing stubs from solutions, never overwrites", "author stubs generate"],
      ["study site", "Rebuild these offline HTML notes", "docs html notes site"]
    ];
    var grid = $(".cmdgrid", w), input = $(".search", w);
    function draw() {
      var q = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var hits = cmds.filter(function (c) { var hay = (c[0] + " " + c[1] + " " + c[2]).toLowerCase(); return q.every(function (t) { return hay.indexOf(t) > -1; }); });
      grid.innerHTML = hits.length ? hits.map(function (c, i) {
        return '<div class="cmd"><code>' + esc(c[0]) + '</code><p>' + esc(c[1]) + '</p><button type="button" class="copy inline" data-i="' + i + '">Copy</button></div>';
      }).join("") : '<p class="muted">No command matches. Try "quiz" or "weak".</p>';
      $$(".copy", grid).forEach(function (b) { b.addEventListener("click", function () { copy(hits[+b.dataset.i][0].split(" / ")[0].replace(/ \[.*?\]/g, ""), b); }); });
    }
    input.addEventListener("input", draw); draw();
  });

  /* ---------- spot the bad test ---------- */
  $$('[data-widget="badtest"]').forEach(function (w) {
    $$(".bt", w).forEach(function (t) {
      var bar = document.createElement("div"); bar.className = "bt-ask";
      bar.innerHTML = '<span>Wrong result on a stub?</span><button type="button" data-a="yes">Yes</button><button type="button" data-a="no">No, shows TODO</button><p class="bt-fb" aria-live="polite"></p>';
      t.appendChild(bar);
      $$("button", bar).forEach(function (b) {
        b.addEventListener("click", function () {
          var guessBad = b.dataset.a === "yes", bad = t.dataset.bad === "true", ok = guessBad === bad, fb = $(".bt-fb", bar);
          fb.className = "bt-fb " + (ok ? "right" : "wrong");
          fb.innerHTML = "<strong>" + (ok ? "Right. " : "Not quite. ") + (bad ? "This is a bad test." : "This is a good test.") + "</strong> " + t.dataset.why;
        });
      });
    });
  });

  /* ---------- chunker playground ---------- */
  $$('[data-widget="chunker"]').forEach(function (w) {
    var ta = $(".ta", w), out = $(".chunks", w), sum = $("[data-summary]", w), size = $('[data-in="size"]', w), ov = $('[data-in="overlap"]', w);
    function draw() {
      var s = +size.value, o = Math.min(+ov.value, s - 1), text = ta.value, step = s - o, chunks = [];
      $('[data-o="size"]', w).textContent = s; $('[data-o="overlap"]', w).textContent = o;
      for (var i = 0, idx = 0; i < text.length; i += step, idx++) {
        var end = Math.min(i + s, text.length); chunks.push({ i: idx, start: i, end: end, text: text.slice(i, end) });
        if (end === text.length) break;
      }
      var dup = chunks.slice(1).reduce(function (a, c) { return a + Math.min(o, c.text.length); }, 0);
      sum.textContent = chunks.length + " chunk" + (chunks.length === 1 ? "" : "s") + ", " + text.length + " characters, " + dup + " repeated through overlap" + (text.length ? " (" + Math.round(dup / text.length * 100) + "% extra to embed and store)" : "") + ".";
      out.innerHTML = chunks.map(function (c, k) {
        var head = k ? Math.min(o, c.text.length) : 0;
        return '<div class="chunk"><div class="chunk-h"><b>#' + k + '</b><span class="chunk-m">chars ' + c.start + "–" + c.end + '</span></div><p>' +
          (head ? "<mark>" + esc(c.text.slice(0, head)) + "</mark>" : "") + esc(c.text.slice(head)) + "</p></div>";
      }).join("");
    }
    [ta, size, ov].forEach(function (e) { e.addEventListener("input", draw); }); draw();
  });

  /* ---------- decision form ---------- */
  $$('[data-widget="decision"]').forEach(function (w) {
    var out = $("[data-out]", w), f = {};
    function draw() {
      $$("[data-f]", w).forEach(function (i) { f[i.dataset.f] = i.value.trim(); });
      var d = new Date(), pad = function (n) { return (n < 10 ? "0" : "") + n; };
      out.textContent = "## " + d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + " - " + (f.title || "<decision>") +
        "\n- **Context:** " + (f.context || "...") + "\n- **Options:** " + (f.options || "A, B, C") + "\n- **Chose:** " + (f.chose || "...") +
        "\n- **Trade-off accepted:** " + (f.tradeoff || "...") + "\n- **Would revisit if:** " + (f.revisit || "...") + "\n";
    }
    $$("[data-f]", w).forEach(function (i) { i.addEventListener("input", draw); }); draw();
    $("[data-copy]", w).addEventListener("click", function (e) { copy(out.textContent, e.target); });
  });

  /* ---------- lazy pipeline stepper ---------- */
  $$('[data-widget="lazy"]').forEach(function (w) {
    var lines = ["# header", "INFO  service started", "", "error disk full", "WARN  slow request", "ERROR oom killed", "INFO  bye", "ERROR never read"];
    var events = [], read = [], got = 0;
    function build() {
      events = []; var i = 0, found = 0;
      function pull() { var out = null;
        events.push({ k: "take", m: "take asks only() for item " + (found + 1) });
        events.push({ k: "only", m: "only asks parse() for the next record" });
        while (out === null && i < lines.length) {
          var ln = lines[i++]; events.push({ k: "src", i: i - 1, m: "parse reads line " + (i - 1) + ": " + JSON.stringify(ln) });
          var t = ln.trim();
          if (!t || t[0] === "#") { events.push({ k: "skip", i: i - 1, m: "blank or comment, skipped, parse loops" }); continue; }
          var lvl = t.split(/\s+/)[0].toUpperCase();
          if (lvl === "ERROR") { out = ln; events.push({ k: "yield", i: i - 1, m: "level ERROR passes the filter, item " + (found + 1) + " is yielded" }); }
          else events.push({ k: "drop", i: i - 1, m: "level " + lvl + " rejected by only(), pull again" });
        }
        return out;
      }
      while (found < 2) { if (pull() !== null) found++; else break; }
      events.push({ k: "done", m: "take has 2 items, so it stops without touching the remaining lines" });
    }
    var pos = 0, ol = $(".lz-lines", w), log = $(".lz-log", w), state = {};
    function draw() {
      state = {}; log.innerHTML = ""; var readN = 0;
      events.slice(0, pos).forEach(function (e) {
        if (e.i !== undefined) { state[e.i] = e.k; if (e.k === "src") readN = Math.max(readN, e.i + 1); }
        var d = document.createElement("div"); d.className = "ev " + e.k; d.textContent = e.m; log.appendChild(d);
      });
      log.scrollTop = log.scrollHeight;
      ol.innerHTML = lines.map(function (l, i) { return '<li class="' + (state[i] || "") + '"><code>' + (esc(l) || "&nbsp;") + "</code></li>"; }).join("");
      var n = Object.keys(state).length;
      $("[data-count]", w).textContent = "Lines read: " + n + " of " + lines.length;
      $("[data-note]", w).textContent = pos >= events.length ? "Done after reading " + n + " of " + lines.length + " lines. A list-based version would read all " + lines.length + " and build three intermediate lists first." : "";
      $("[data-step]", w).disabled = pos >= events.length;
    }
    $("[data-step]", w).addEventListener("click", function () { if (pos < events.length) pos++; draw(); });
    $("[data-reset]", w).addEventListener("click", function () { pos = 0; draw(); });
    build(); draw();
  });

  /* ---------- async timeline ---------- */
  $$('[data-widget="timeline"]').forEach(function (w) {
    var dur = [1.2, 0.6, 1.8, 0.9, 1.4, 0.7, 1.1, 1.6], mode = "seq", gantt = $(".gantt", w), limit = $('[data-in="limit"]', w);
    function schedule() {
      var L = +limit.value, starts = [], free = new Array(Math.min(L, 8)).fill(0);
      dur.forEach(function (d, i) {
        if (mode === "seq") starts.push(i ? starts[i - 1] + dur[i - 1] : 0);
        else if (mode === "gather") starts.push(0);
        else { var k = free.indexOf(Math.min.apply(null, free)); starts.push(free[k]); free[k] += d; }
      });
      return starts;
    }
    function draw() {
      $('[data-o="limit"]', w).textContent = limit.value; $("[data-limit-wrap]", w).hidden = mode !== "sem";
      var st = schedule(), end = Math.max.apply(null, st.map(function (s, i) { return s + dur[i]; })), sumDur = dur.reduce(function (a, b) { return a + b; }, 0);
      gantt.innerHTML = dur.map(function (d, i) {
        return '<div class="grow"><span>call ' + (i + 1) + '</span><div class="gtrack"><i title="' + d + 's" style="left:' + (st[i] / sumDur * 100) + '%;width:' + (d / sumDur * 100) + '%">' + (d / sumDur * 100 >= 9 ? d + "s" : "") + '</i></div></div>';
      }).join("") + '<div class="gaxis"><span>0s</span><span>' + sumDur.toFixed(1) + 's</span></div>';
      var note = { seq: "Each call waits for the previous one. Total is the sum of every duration.", gather: "All eight start together, so total is the longest single call. Fine for 8 calls, dangerous for 8,000 against a rate-limited API.", sem: "At most " + limit.value + " run at once. A finished call frees a slot for the next one, so you get most of the speedup without flooding the API." }[mode];
      $("[data-note]", w).innerHTML = "<strong>Total " + end.toFixed(1) + "s</strong> (sequential would take " + sumDur.toFixed(1) + "s). " + note;
    }
    $$("[data-mode]", w).forEach(function (b) { b.addEventListener("click", function () { mode = b.dataset.mode; $$("[data-mode]", w).forEach(function (x) { x.classList.toggle("on", x === b); }); draw(); }); });
    limit.addEventListener("input", draw); draw();
  });

  /* ---------- backoff visualiser ---------- */
  $$('[data-widget="backoff"]').forEach(function (w) {
    var draws = [], bars = $(".bars", w);
    function params() { var p = {}; $$("[data-in]", w).forEach(function (i) { p[i.dataset.in] = +i.value; }); $('[data-o="base"]', w).textContent = p.base; $('[data-o="max"]', w).textContent = p.max; $('[data-o="n"]', w).textContent = p.n; $('[data-o="ra"]', w).textContent = p.ra ? p.ra + " s" : "off"; return p; }
    function roll() { draws = []; for (var i = 0; i < 10; i++) draws.push(Math.random()); }
    function draw() {
      var p = params(), top = Math.max(p.max, p.ra, 1), total = 0, html = "";
      for (var a = 0; a < p.n; a++) {
        var cap = Math.min(p.max, p.base * Math.pow(2, a)), wait = p.ra ? p.ra : draws[a] * cap; total += wait;
        html += '<div class="brow"><span>retry ' + (a + 1) + '</span><div class="btrack"><i class="cap" style="width:' + (cap / top * 100) + '%"></i><i class="val' + (p.ra ? " ra" : "") + '" style="width:' + (wait / top * 100) + '%"></i></div><em>' + wait.toFixed(2) + "s</em></div>";
      }
      bars.innerHTML = html; $("[data-total]", w).textContent = "Total waiting: " + total.toFixed(1) + "s across " + p.n + " retries";
    }
    $$("[data-in]", w).forEach(function (i) { i.addEventListener("input", draw); });
    $("[data-resample]", w).addEventListener("click", function () { roll(); draw(); }); roll(); draw();
  });

  /* ---------- predict deck filter ---------- */
  $$('[data-widget="predict"]').forEach(function (w) {
    var sel = $("select", w), items = $$(".pq", w);
    sel.addEventListener("change", function () { items.forEach(function (i) { i.hidden = sel.value && i.dataset.module !== sel.value; }); });
    $("[data-open]", w).addEventListener("click", function () { var open = items.some(function (i) { return !i.hidden && !i.open; }); items.forEach(function (i) { if (!i.hidden) i.open = open; }); });
  });

  /* ---------- roadmap tracker ---------- */
  (function () {
    var boxes = $$(".prose .box"); if (!boxes.length || !$('[data-widget="tracker"]')) return;
    var saved = store("notes-roadmap") || {};
    var groups = {}, order = [];
    function groupOf(b) {
      var el = b.closest(".prose > *"); // the top-level block holding this box
      while (el && el.tagName !== "H2") el = el.previousElementSibling;
      return el ? el.textContent.replace(/\s*\(.*$/, "").replace(/:.*$/, "").trim() : "Other";
    }
    boxes.forEach(function (b, i) {
      var title = groupOf(b), row = b.closest("li, tr") || b;
      var label = row.textContent.replace(/\s+/g, " ").trim().slice(0, 60);
      var input = document.createElement("input"); input.type = "checkbox"; input.className = "rm"; input.id = "rm-" + i;
      input.setAttribute("aria-label", label); input.checked = !!saved[label];
      b.replaceWith(input);
      input.addEventListener("change", function () { saved[label] = input.checked; store("notes-roadmap", saved); sumup(); });
      if (!groups[title]) { groups[title] = []; order.push(title); }
      groups[title].push(input);
    });
    var host = $('[data-widget="tracker"] .tr-rows');
    function sumup() {
      var all = 0, done = 0;
      host.innerHTML = order.map(function (t) {
        var d = groups[t].filter(function (i) { return i.checked; }).length, n = groups[t].length; all += n; done += d;
        return '<div class="tr-row"><span>' + esc(t) + '</span><div class="tr-track"><i style="width:' + (n ? d / n * 100 : 0) + '%"></i></div><em>' + d + "/" + n + "</em></div>";
      }).join("");
      $("[data-tr-total]").textContent = done + " of " + all + " milestones ticked";
    }
    sumup();
    $("[data-tr-reset]").addEventListener("click", function () { Object.keys(groups).forEach(function (t) { groups[t].forEach(function (i) { i.checked = false; }); }); saved = {}; store("notes-roadmap", saved); sumup(); });
  })();
})();
