import { useEffect, useRef, useState } from "react";

const example = {
  name: "项目周报",
  sources: "notes.md",
  target: "weekly-brief.md",
  sections: "本周事项 | TODO:",
};

function fromRecipe(recipe) {
  return {
    name: recipe.name,
    sources: recipe.sources.join("\n"),
    target: recipe.target,
    sections: recipe.sections.map(item => `${item.heading} | ${item.prefix}`).join("\n"),
  };
}

function splitLines(value, maximum) {
  const result = value.split(/\r?\n/).map(item => item.trim()).filter(Boolean);
  if (!result.length || result.length > maximum) throw new Error(`请填写 1–${maximum} 行。`);
  return result;
}

export default function ReportRecipes() {
  const [recipes, setRecipes] = useState([]);
  const [selected, setSelected] = useState("");
  const [form, setForm] = useState(example);
  const [preview, setPreview] = useState(null);
  const [discovery, setDiscovery] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const inFlight = useRef(false);

  function current(recipes, id) { return recipes.find(recipe => recipe.id === id); }
  const saved = current(recipes, selected);
  const dirty = saved && JSON.stringify(fromRecipe(saved)) !== JSON.stringify(form);

  function edit(key, value) {
    setForm({ ...form, [key]: value });
    setPreview(null);
    if (key === "sources") setDiscovery(null);
  }

  async function act(operation, fields = {}) {
    if (inFlight.current) return null;
    inFlight.current = true;
    setBusy(true);
    setStatus("");
    try {
      const result = await callAction({
        name: "report_recipe",
        payload: { operation, ...fields },
      });
      const data = result?.response ?? result;
      if (!data?.ok) throw new Error(data?.message || "操作失败，请重试。");
      if (data.recipes) setRecipes(data.recipes);
      if (data.message) setStatus(data.message);
      return data;
    } catch (error) {
      setStatus(error.message || "连接失败，请重试。");
      return null;
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  useEffect(() => { act("refresh"); }, []);

  function choose(id) {
    setSelected(id);
    setPreview(null);
    setDiscovery(null);
    const recipe = current(recipes, id);
    setForm(recipe ? fromRecipe(recipe) : example);
  }

  function fields() {
    const sections = splitLines(form.sections, 6).map((line, index) => {
      const separator = line.indexOf("|");
      if (separator < 1) throw new Error(`第 ${index + 1} 个章节须写成「标题 | 行前缀」。`);
      return {
        heading: line.slice(0, separator).trim(),
        prefix: line.slice(separator + 1).trim(),
      };
    });
    return {
      name: form.name,
      sources: splitLines(form.sources, 8),
      target: form.target,
      sections,
    };
  }

  async function save(event) {
    event.preventDefault();
    try {
      const data = await act(selected ? "save" : "create", {
        id: selected || undefined,
        ...fields(),
      });
      if (data) {
        if (data.id) setSelected(data.id);
        const savedRecipe = current(data.recipes, data.id);
        if (savedRecipe) setForm(fromRecipe(savedRecipe));
        setPreview(null);
      }
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function makePreview() {
    if (!selected) {
      setStatus("先保存方案，再从当前文件生成预览。");
      return;
    }
    const data = await act("preview", { id: selected });
    if (data?.preview) setPreview({ ...data.preview, token: data.token });
  }

  async function inspect() {
    setDiscovery(null);
    try {
      const data = await act("inspect", { sources: splitLines(form.sources, 8) });
      if (data?.discovery) setDiscovery(data.discovery);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function addMarker(marker) {
    try {
      const sections = form.sections.trim() ? fields().sections : [];
      if (sections.length >= 6) throw new Error("最多 6 个章节，请先移除不需要的章节。");
      if (sections.some(item => item.prefix.startsWith(marker.prefix) || marker.prefix.startsWith(item.prefix))) {
        throw new Error("此标记与现有前缀重复或重叠，请直接编辑章节配置。");
      }
      edit("sections", [...sections, marker].map(item => `${item.heading} | ${item.prefix}`).join("\n"));
      setStatus("已加入章节配置；保存后可预览当前资料。");
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function apply() {
    const pending = preview;
    setPreview(null);
    await act("apply", { id: selected, token: pending.token });
  }

  return <section className="harness-panel harness-recipe-panel" aria-label="个人任务方案">
    <h2>个人任务方案</h2>
    <div className="harness-recipe-actions">
      <label>我的方案
        <select value={selected} onChange={event => choose(event.target.value)} disabled={busy}>
          <option value="">新方案</option>
          {recipes.map(recipe => <option key={recipe.id} value={recipe.id}>{recipe.name}</option>)}
        </select>
      </label>
      {selected && <button type="button" disabled={busy} onClick={async () => {
        if (!window.confirm("删除此方案及方案内的执行统计？聊天用量记录和已生成的简报文件不受影响。")) return;
        const data = await act("delete", { id: selected });
        if (data) { setSelected(""); setForm(example); setPreview(null); setDiscovery(null); }
      }}>删除方案</button>}
    </div>
    {selected && <p className="harness-recipe-stats">
      本机完成 {saved?.runs ?? 0} 次 · 最近完成 {saved?.last_run ? new Date(saved.last_run).toLocaleString() : "暂无"}
      · 每次 0 次模型请求 · API 费用 ¥0
    </p>}
    <form className="harness-recipe-form" onSubmit={save}>
      <label>方案名称<input required maxLength={80} value={form.name}
        disabled={busy} onChange={event => edit("name", event.target.value)} /></label>
      <label>来源文件（相对工作区路径，每行一个，最多 8 个）
        <textarea required maxLength={2000} rows={3} value={form.sources}
          disabled={busy} onChange={event => edit("sources", event.target.value)} />
      </label>
      <div>
        <button type="button" disabled={busy || !form.sources.trim()} onClick={inspect}>从来源查找可用标记</button>
        <p>本地读取指定文件，列出行首标签和 Markdown 待办；选择后加入章节配置。0 次模型请求。</p>
      </div>
      {discovery && <div className="harness-marker-list" aria-label="来源中的可用标记">
        <p>已完整读取 {discovery.source_count} 份来源。这里只按原文标记整理，不改写或推断内容。</p>
        {!discovery.candidates.length && <p>未找到可用标记。可在资料中使用「TODO: 事项」「风险：说明」或「- [ ] 事项」，也可手动填写精确前缀。</p>}
        {discovery.candidates.map(marker => <div className="harness-marker" key={marker.prefix}>
          <button type="button" disabled={busy} onClick={() => addMarker(marker)}>加入「{marker.prefix}」</button>
          <span>{marker.count} 处 · {marker.path}:{marker.line}</span>
          <p>{marker.example}{marker.example_truncated ? "…" : ""}</p>
        </div>)}
        {discovery.additional_markers_omitted && <p>最多列出 {discovery.candidate_limit} 种标记，其余未展示。可缩小来源，或手动填写需要的前缀。</p>}
      </div>}
      <label>每行章节配置为「标题 | 精确前缀」（最多 6 个章节）
        <textarea required maxLength={2000} rows={3} value={form.sections}
          disabled={busy} onChange={event => edit("sections", event.target.value)} />
      </label>
      <label>Markdown 输出文件（相对工作区路径）<input required maxLength={240}
        disabled={busy} value={form.target} onChange={event => edit("target", event.target.value)} />
      </label>
      <div className="harness-recipe-actions">
        <button type="submit" disabled={busy || (saved && !dirty)}>保存个人方案</button>
        <button type="button" disabled={busy || !selected || dirty} onClick={makePreview}>读取当前资料并预览</button>
      </div>
      {dirty && <p>有未保存的修改。保存后即可重新预览。</p>}
    </form>
    {preview && <div className="harness-recipe-preview">
      <h3>写入前预览 · {preview.path}</h3>
      <p>{preview.source_count} 份已完整读取的来源 · 0 次模型请求 · API 费用 ¥0
        {preview.unchanged ? " · 文件内容没有变化" : ""}</p>
      <details open>
        <summary>完整简报内容</summary><pre>{preview.content}</pre>
      </details>
      {preview.diff_truncated && <p>差异较长，以下仅展示部分差异。请核对上方完整简报。</p>}
      <details>
        <summary>与现有文件的差异</summary><pre>{preview.diff || "没有变化"}</pre>
      </details>
      <button type="button" disabled={busy || preview.unchanged} onClick={apply}>
        确认写入
      </button>
      <button type="button" disabled={busy} onClick={() => setPreview(null)}>取消预览</button>
    </div>}
    <p className="harness-status" role="status">{status}</p>
  </section>;
}
