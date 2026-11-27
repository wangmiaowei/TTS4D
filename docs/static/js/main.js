(function () {
  "use strict";
  const C = window.TTS_CONFIG || {};
  const CASES = window.TTS_CASES || [];
  const media = (p) => C.mediaBase + p;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  /* ---------- header links ---------- */
  function setLink(id, url) {
    const a = document.getElementById(id);
    if (!a) return;
    if (url) a.href = url; else { a.classList.add("disabled"); a.title = "coming soon"; }
  }
  setLink("lnk-paper", C.arxiv);
  {
    const w = $("#yt-wrap");
    if (C.youtubeId) {
      w.innerHTML = '<iframe src="https://www.youtube-nocookie.com/embed/' + C.youtubeId +
        '?rel=0" title="Results video" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen loading="lazy"></iframe>';
    } else {
      // self-hosted: the narrated results video streamed from the media repository (metadata only until play)
      w.innerHTML = '<video controls playsinline preload="none" poster="' + media("video/poster.jpg") + '" src="' +
        media("video/results.mp4") + '"></video>';
    }
  }

  /* ---------- hero canvas: tokens flowing into a fixed-size memory ---------- */
  (function hero() {
    const cv = $("#hero-canvas");
    if (!cv || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const ctx = cv.getContext("2d");
    let W, H, pts = [];
    const N = 90;
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function spawn(p) {
      p.x = Math.random() * W; p.y = Math.random() * H; p.vx = (Math.random() - .5) * .25; p.vy = (Math.random() - .5) * .25;
      p.r = 1 + Math.random() * 1.8; p.h = Math.random() < .5 ? 210 : 265;
      return p;
    }
    resize(); window.addEventListener("resize", resize);
    for (let i = 0; i < N; i++) pts.push(spawn({}));
    let visible = true;
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(cv);
    function frame() {
      if (visible) {
        ctx.clearRect(0, 0, W, H);
        for (let i = 0; i < N; i++) {
          const p = pts[i];
          p.x += p.vx; p.y += p.vy;
          if (p.x < -10 || p.x > W + 10 || p.y < -10 || p.y > H + 10) spawn(p);
          for (let j = i + 1; j < N; j++) {
            const q = pts[j], dx = p.x - q.x, dy = p.y - q.y, d = dx * dx + dy * dy;
            if (d < 13000) {
              ctx.strokeStyle = "hsla(" + p.h + ",90%,70%," + (0.16 * (1 - d / 13000)) + ")";
              ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
            }
          }
          ctx.fillStyle = "hsla(" + p.h + ",95%,75%,.85)";
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fill();
        }
      }
      requestAnimationFrame(frame);
    }
    frame();
  })();

  /* ---------- counters + reveal ---------- */
  const io = new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      $$(".num[data-count]", e.target).forEach((n) => {
        const target = parseFloat(n.dataset.count), dec = parseInt(n.dataset.dec || "0", 10), t0 = performance.now();
        (function tick(t) {
          const k = Math.min(1, (t - t0) / 1200), v = target * (1 - Math.pow(1 - k, 3));
          n.textContent = v.toFixed(dec);
          if (k < 1) requestAnimationFrame(tick);
        })(t0);
      });
      io.unobserve(e.target);
    });
  }, { threshold: 0.15 });
  $$(".section .container > *").forEach((el) => { el.classList.add("reveal"); io.observe(el); });
  $$(".stats").forEach((el) => io.observe(el));

  /* ---------- synchronized video group ---------- */
  function Group(master, others) {
    this.master = master; this.others = others;
    const self = this;
    this.timer = setInterval(function () {
      if (!self.master || self.master.paused) return;
      const t = self.master.currentTime;
      self.others.forEach((v) => {
        if (v.readyState >= 2 && v.duration && Math.abs(v.currentTime - t) > 0.12) v.currentTime = Math.min(t, v.duration - 0.05);
      });
    }, 400);
  }
  function playAll(vs) { vs.forEach((v) => { const p = v.play(); if (p && p.catch) p.catch(() => {}); }); }

  /* ---------- comparison panel ---------- */
  const METHODS = [
    ["FSM", "FSM", "ECCV'26"], ["FL4N", "FL4N", "arXiv'26"], ["MoVieS", "MoVieS", "CVPR'26"],
    ["4DGT", "4DGT", "NeurIPS'25"], ["NeoVerse", "NeoVerse", "CVPR'26"], ["C4G", "C4G†", "arXiv'26"]
  ];
  const INITIAL = 6;
  let curCase = null, curMethod = "FSM", mode = "side", showAll = false, curFilter = "All";
  const grid = $("#case-grid"), filters = $("#ds-filters"), moreBtn = $("#btn-more");
  const V = (role) => $('video[data-role="' + role + '"]');
  const roles = ["input", "camera", "unc", "ours", "base", "swipe-base", "swipe-ours"];
  let group = null;

  const groups = ["All"].concat(Array.from(new Set(CASES.map((c) => c.group))));
  groups.forEach((g) => {
    const b = document.createElement("button");
    b.className = "chip" + (g === "All" ? " active" : ""); b.textContent = g;
    b.onclick = () => { curFilter = g; $$(".chip", filters).forEach((x) => x.classList.toggle("active", x === b)); renderGrid(); };
    filters.appendChild(b);
  });

  function renderGrid() {
    grid.innerHTML = "";
    const list = CASES.filter((c) => curFilter === "All" || c.group === curFilter);
    const phone = window.innerWidth < 720;          // phones: every scene in one swipeable strip, no "More cases" button
    list.forEach((c, i) => {
      const d = document.createElement("div");
      d.className = "case" + (curCase && curCase.id === c.id ? " active" : "");
      d.hidden = !phone && !showAll && curFilter === "All" && i >= INITIAL;
      d.innerHTML = '<img ' + (i < 6 ? '' : 'loading="lazy" ') + 'alt="" src="' + media("cmp/" + c.id + "/poster.jpg") + '"><div class="cap">' + c.title +
        '<small>' + c.dataset + ' · ' + c.nin + ' views</small></div>';
      d.onclick = () => selectCase(c, true);
      grid.appendChild(d);
    });
    const hidden = phone ? 0 : list.filter((_, i) => !showAll && curFilter === "All" && i >= INITIAL).length;
    moreBtn.hidden = phone || (hidden === 0 && !showAll);
    moreBtn.firstChild.textContent = showAll ? "Fewer cases " : "More cases ";
    $("#more-count").textContent = showAll ? "" : "(" + hidden + ")";
  }
  moreBtn.onclick = () => { showAll = !showAll; renderGrid(); };

  const bar = $("#method-bar");
  bar.innerHTML = '<span class="lbl">Compare Ours with:</span>';
  METHODS.forEach(([key, label, venue]) => {
    const b = document.createElement("button");
    b.className = "mbtn" + (key === curMethod ? " active" : ""); b.dataset.m = key;
    b.innerHTML = label + "<small>" + venue + "</small>";
    b.onclick = () => { curMethod = key; $$(".mbtn", bar).forEach((x) => x.classList.toggle("active", x === b)); loadBaseline(); };
    bar.appendChild(b);
  });

  function setSrc(v, url) { if (v.getAttribute("src") !== url) { v.src = url; v.load(); } }

  function loadBaseline() {
    if (!curCase) return;
    const label = METHODS.find((m) => m[0] === curMethod)[1];
    $("#bl-lbl").textContent = label; $("#swipe-bl-lbl").textContent = label;
    const url = media("cmp/" + curCase.id + "/" + curMethod + ".mp4");
    const tgt = mode === "side" ? V("base") : V("swipe-base");
    const t = V(mode === "side" ? "ours" : "swipe-ours").currentTime || 0;
    setSrc(tgt, url);
    tgt.addEventListener("loadeddata", function once() { tgt.removeEventListener("loadeddata", once); try { tgt.currentTime = t; } catch (e) {} if (!playing) return; playAll([tgt]); });
  }

  let playing = true;
  function activeVideos() {
    const top = [V("input"), V("camera"), V("unc")];
    return mode === "side" ? top.concat([V("ours"), V("base")]) : top.concat([V("swipe-ours"), V("swipe-base")]);
  }
  function selectCase(c, scroll) {
    curCase = c;
    $$(".case", grid).forEach((d, i) => d.classList.toggle("active", false));
    renderGrid();
    const st = $("#stage");
    st.style.setProperty("--ar", c.ar || 16 / 9);
    layout();
    $("#v-title").textContent = c.title;
    $("#v-sub").textContent = c.dataset + " · " + c.nin + " posed input views · novel space-time camera path";
    $$(".mbtn", bar).forEach((b) => { b.disabled = c.videos.indexOf(b.dataset.m) < 0; });
    if (c.videos.indexOf(curMethod) < 0) { curMethod = "FSM"; $$(".mbtn", bar).forEach((x) => x.classList.toggle("active", x.dataset.m === "FSM")); }
    setSrc(V("input"), media("cmp/" + c.id + "/input.mp4"));
    setSrc(V("camera"), media("cmp/" + c.id + "/camera.mp4"));
    setSrc(V("unc"), media("cmp/" + c.id + "/Ours_unc.mp4"));
    const oursRole = mode === "side" ? "ours" : "swipe-ours";
    setSrc(V(oursRole), media("cmp/" + c.id + "/Ours.mp4"));
    loadBaseline();
    if (group) clearInterval(group.timer);
    const vs = activeVideos();
    group = new Group(vs[3], vs.filter((v, i) => i !== 3));
    playing = true; $("#btn-play").textContent = "❚❚";
    playAll(vs);
    if (scroll) $("#viewer").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $$(".mode-toggle button").forEach((b) => {
    b.onclick = () => {
      mode = b.dataset.mode;
      $$(".mode-toggle button").forEach((x) => x.classList.toggle("active", x === b));
      $("#main-side").hidden = mode !== "side"; $("#main-swipe").hidden = mode !== "swipe";
      [V("ours"), V("base"), V("swipe-ours"), V("swipe-base")].forEach((v) => v.pause());
      if (curCase) selectCase(curCase, false);
    };
  });

  /* band layout: Input/Camera (stacked) | method | Ours | uncertainty, sized to fit the width and the screen height */
  function layout() {
    const stage = $("#stage"), band = $("#band");
    const ar = (curCase && curCase.ar) || 16 / 9, gap = 10, W = stage.clientWidth;
    const narrow = W < 720;
    band.classList.toggle("narrow", narrow);
    let h, hs;
    if (narrow) { h = Math.min((W - gap) / 2 / ar, window.innerHeight * 0.4); hs = h; }
    else {
      h = (W - 3 * gap + gap * ar / 2) / (3.5 * ar);
      h = Math.max(90, Math.min(h, window.innerHeight * 0.55));
      hs = (h - gap) / 2;
    }
    stage.style.setProperty("--h", h + "px");
    stage.style.setProperty("--hs", hs + "px");
    stage.style.setProperty("--sw", (2 * h * ar + gap) + "px");
    // swipe view: one frame with the video's own aspect ratio, as wide as the two side-by-side panes (taller if needed)
    const slotW = narrow ? W : 2 * h * ar + gap;
    const swh = Math.min(slotW / ar, window.innerHeight * 0.5), sww = swh * ar;
    stage.style.setProperty("--swh", swh + "px");
    stage.style.setProperty("--sww", sww + "px");
  }
  window.addEventListener("resize", layout);
  layout();

  /* swipe */
  (function swipe() {
    const wrap = $("#main-swipe"), top = $("#swipe-top"), h = $("#swipe-handle");
    let drag = false;
    function at(x) {
      const r = wrap.getBoundingClientRect(), f = Math.max(0, Math.min(1, (x - r.left) / r.width));
      top.style.clipPath = "inset(0 " + ((1 - f) * 100) + "% 0 0)"; h.style.left = f * 100 + "%";
    }
    wrap.addEventListener("pointerdown", (e) => { drag = true; wrap.setPointerCapture(e.pointerId); at(e.clientX); });
    wrap.addEventListener("pointermove", (e) => { if (drag) at(e.clientX); });
    wrap.addEventListener("pointerup", () => { drag = false; });
  })();

  /* transport */
  $("#btn-play").onclick = () => {
    const vs = activeVideos();
    playing = !playing;
    if (playing) playAll(vs); else vs.forEach((v) => v.pause());
    $("#btn-play").textContent = playing ? "❚❚" : "▶";
  };
  const seek = $("#seek");
  seek.oninput = () => {
    const vs = activeVideos(), m = vs[3];
    if (!m.duration) return;
    const t = (seek.value / 1000) * m.duration;
    vs.forEach((v) => { try { v.currentTime = Math.min(t, (v.duration || t) - 0.05); } catch (e) {} });
  };
  setInterval(() => { const m = activeVideos()[3]; if (m && m.duration && document.activeElement !== seek) seek.value = (m.currentTime / m.duration) * 1000; }, 250);
  $("#speed").onchange = (e) => activeVideos().forEach((v) => (v.playbackRate = parseFloat(e.target.value)));

  renderGrid();
  // Preselect the first case (loads only when the viewer scrolls into view).
  if (CASES.length) {
    const vo = new IntersectionObserver((e) => { if (e[0].isIntersecting && !curCase) { selectCase(CASES[0], false); vo.disconnect(); } }, { threshold: 0.2 });
    vo.observe($("#viewer"));
  }

  /* ---------- robustness ---------- */
  const resPsnr = { fsm_256: 18.60, fsm_384: 18.21, fsm_512: 18.24, ours_256: 27.81, ours_384: 28.05, ours_512: 27.04 };
  let viewsN = 64, resR = 512;
  const robLoaded = {};
  function loadViews() {
    const a = $("#rv-fsm"), b = $("#rv-ours");
    setSrc(a, media("views/fsm_" + viewsN + ".mp4")); setSrc(b, media("views/ours_" + viewsN + ".mp4"));
    if (!a._g) a._g = new Group(b, [a]);
    playAll([a, b]);
  }
  function loadRes() {
    const a = $("#rr-fsm"), b = $("#rr-ours");
    setSrc(a, media("res/fsm_" + resR + ".mp4")); setSrc(b, media("res/ours_" + resR + ".mp4"));
    $("#rr-fsm-p").textContent = resPsnr["fsm_" + resR].toFixed(1) + " dB"; $("#rr-ours-p").textContent = resPsnr["ours_" + resR].toFixed(1) + " dB";
    if (!a._g) a._g = new Group(b, [a]);
    playAll([a, b]);
  }
  function loadAvs() {
    const vs = $$(".avs-grid video");
    vs.forEach((v) => setSrc(v, media(v.dataset.src)));
    playAll(vs);
  }
  const loaders = { views: loadViews, res: loadRes, avs: loadAvs };
  // load each block the first time it scrolls into view; pause it while it is off screen
  $$(".btn-load").forEach((b) => {
    const key = b.dataset.load, card = b.closest(".card");
    b.classList.add("done");
    new IntersectionObserver((ents) => {
      const vs = $$("video", card);
      if (ents[0].isIntersecting) {
        if (!robLoaded[key]) { robLoaded[key] = true; loaders[key](); } else playAll(vs);
      } else vs.forEach((v) => v.pause());
    }, { threshold: 0.2 }).observe(card);
  });
  $$("#seg-views button").forEach((b) => b.onclick = () => {
    viewsN = +b.dataset.v; $$("#seg-views button").forEach((x) => x.classList.toggle("active", x === b));
    if (robLoaded.views) loadViews();
  });
  $$("#seg-res button").forEach((b) => b.onclick = () => {
    resR = +b.dataset.r; $$("#seg-res button").forEach((x) => x.classList.toggle("active", x === b));
    if (robLoaded.res) loadRes();
  });

  /* ---------- results reel: load and play only while visible ---------- */
  const reelVids = $$("#reel video");
  reelVids.forEach((v) => { v.poster = media(v.dataset.src.replace("Ours.mp4", "poster.jpg")); });
  const reelIO = new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting) { if (!v.getAttribute("src")) v.src = media(v.dataset.src); playAll([v]); }
      else v.pause();
    });
  }, { threshold: 0.25 });
  reelVids.forEach((v) => reelIO.observe(v));

  /* ---------- sticky nav: show after the hero, highlight the current section ---------- */
  const nav = $("#topnav"), hero = $(".hero");
  new IntersectionObserver((e) => nav.classList.toggle("show", !e[0].isIntersecting), { threshold: 0.05 }).observe(hero);
  const navLinks = $$(".tn-links a");
  const secIO = new IntersectionObserver((ents) => {
    ents.forEach((e) => {
      if (!e.isIntersecting) return;
      navLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("main section[id]").forEach((s) => secIO.observe(s));

  /* ---------- bibtex ---------- */
  $("#copy-bib").onclick = (e) => {
    const t = $("#bib-text").textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(
      () => { e.target.textContent = "Copied!"; setTimeout(() => (e.target.textContent = "Copy"), 1500); },
      () => { const r = document.createRange(); r.selectNodeContents($("#bib-text")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); });
  };
})();
