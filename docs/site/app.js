(() => {
  const $ = (s) => document.querySelector(s);
  const pages = [...document.querySelectorAll("[data-page]")];
  const sidebar = $("#sidebar");
  const menu = $("#menu");
  function navigate() {
    const hash = decodeURIComponent(location.hash.slice(1));
    let slug = pages.find(p => p.dataset.page === hash)?.dataset.page;
    let target = null;
    if (!slug && hash) {
      target = document.getElementById(hash);
      slug = target?.closest("[data-page]")?.dataset.page;
    }
    slug ||= "getting-started";
    pages.forEach(p => p.hidden = p.dataset.page !== slug);
    document.querySelectorAll("[data-outline]").forEach(p => p.hidden = p.dataset.outline !== slug);
    document.querySelectorAll("[data-nav]").forEach(a => {
      const active = a.dataset.nav === slug;
      if (active) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    document.title = `${pages.find(p => p.dataset.page === slug).getAttribute("aria-label")} · Personal Harness`;
    sidebar.classList.remove("open"); menu.setAttribute("aria-expanded", "false");
    if (target) target.scrollIntoView(); else window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", navigate); navigate();
  menu.addEventListener("click", () => { const open = sidebar.classList.toggle("open"); menu.setAttribute("aria-expanded", String(open)); });
  $("#theme").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme || "light";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("ph-docs-theme", next); } catch {}
  });
  try { const theme = localStorage.getItem("ph-docs-theme"); if (theme) document.documentElement.dataset.theme = theme; } catch {}
  $("#search").addEventListener("input", e => {
    const query = e.target.value.trim().toLowerCase();
    const result = $("#results"); result.replaceChildren(); result.hidden = !query; $("#nav-items").hidden = !!query;
    if (!query) return;
    const matches = window.DOC_SEARCH.filter(p => `${p.title} ${p.text}`.toLowerCase().includes(query));
    const count = document.createElement("p"); count.textContent = `${matches.length} 篇相关文档`; result.append(count);
    matches.forEach(p => {
      const link = document.createElement("a"); link.href = `#${p.slug}`; link.textContent = p.title;
      link.addEventListener("click", () => { $("#search").value = ""; result.hidden = true; $("#nav-items").hidden = false; sidebar.classList.remove("open"); menu.setAttribute("aria-expanded", "false"); });
      result.append(link);
    });
  });
  document.addEventListener("keydown", e => {
    if (e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      e.preventDefault(); sidebar.classList.add("open"); menu.setAttribute("aria-expanded", "true"); $("#search").focus();
    }
    if (e.key === "Escape") { sidebar.classList.remove("open"); menu.setAttribute("aria-expanded", "false"); $("#search").blur(); }
  });
  document.querySelectorAll("pre").forEach(pre => {
    const button = document.createElement("button"); button.className = "copy"; button.textContent = "复制";
    button.setAttribute("aria-label", "复制代码");
    button.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(pre.querySelector("code").textContent); button.textContent = "已复制"; }
      catch { button.textContent = "请手动复制"; }
      setTimeout(() => button.textContent = "复制", 1800);
    }); pre.append(button);
  });
  const viewer = document.createElement("dialog");
  viewer.className = "image-viewer";
  viewer.setAttribute("aria-label", "截图预览");
  const close = document.createElement("button");
  close.className = "image-close"; close.textContent = "关闭截图 ×";
  close.addEventListener("click", () => viewer.close());
  const expanded = document.createElement("img");
  viewer.append(close, expanded); document.body.append(viewer);
  viewer.addEventListener("click", event => { if (event.target === viewer) viewer.close(); });
  document.querySelectorAll("article img").forEach(img => {
    img.loading = "lazy"; img.decoding = "async";
    const link = document.createElement("a"); link.href = img.getAttribute("src");
    link.className = "screenshot-link"; link.setAttribute("aria-label", `放大截图：${img.alt}`);
    img.replaceWith(link); link.append(img);
    link.addEventListener("click", event => {
      event.preventDefault(); expanded.src = img.src; expanded.alt = img.alt; viewer.showModal();
    });
  });
})();
