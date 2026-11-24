import { useState } from "react";

export default function ModelConnections() {
  const defaults = {name: "", base_url: "", model: "", max_output_tokens: "1600", context_window: "32768", input_per_million_cny: "", output_per_million_cny: "", cache_read_per_million_cny: ""};
  const initial = (props.connections || []).find(c => c.id === props.selected);
  function connectionForm(item) {
    return item ? { ...defaults, ...item.model, name: item.name, cache_read_per_million_cny: item.model.cache_read_per_million_cny ?? "" } : defaults;
  }
  const [form, setForm] = useState(() => connectionForm(initial));
  const [connectionId, setConnectionId] = useState(initial?.id || "");
  const [makeDefault, setMakeDefault] = useState(!props.connections?.length);
  const [key, setKey] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [connections, setConnections] = useState(props.connections || []);
  function choose(id) {
    setConnectionId(id); setKey(""); setStatus("");
    const item = connections.find(c => c.id === id);
    setForm(connectionForm(item));
  }
  async function save(event) {
    event.preventDefault(); setBusy(true); setStatus("保存中…");
    try {
      const result = await callAction({name: "save_model_connection", payload: {connection_id: connectionId || null, ...form, api_key: key, make_default: makeDefault}});
      const response = result?.response ?? result;
      if (response?.connections) setConnections(response.connections);
      if (response?.connection_id) setConnectionId(response.connection_id);
      setKey(""); setStatus(response?.message || "已提交，查看下方保存结果。");
    } catch { setStatus("保存失败，请查看聊天中的提示。"); }
    finally { setBusy(false); }
  }
  async function remove() {
    if (!connectionId || !window.confirm("删除此模型连接？聊天历史仍会保留。")) return;
    setBusy(true);
    try { await callAction({name: "delete_model_connection", payload: {connection_id: connectionId}}); setConnections(connections.filter(c => c.id !== connectionId)); choose(""); setStatus("删除请求已提交。"); }
    catch { setStatus("删除失败。"); } finally { setBusy(false); }
  }
  const field = (name, label, type = "text", required = true) => <label key={name}>{label}<input name={name} type={type} required={required} step="any" min={type === "number" ? "0" : undefined} value={form[name]} onChange={e => setForm({...form, [name]: e.target.value})} /></label>;
  return <section className="harness-panel harness-model-panel" aria-label="模型连接">
    <h2>模型连接</h2>
    <p>保存不发起模型请求。可用 /test-model 测试连接，费用计入预算。</p>
    <div className="harness-model-toolbar">
      <label>编辑连接<select aria-label="编辑连接" value={connectionId} onChange={e => choose(e.target.value)}><option value="">新增连接</option>{connections.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
    </div>
    <form onSubmit={save}>
      <fieldset>
        <legend>连接参数</legend>
        <div className="harness-grid">
          {field("name", "连接名称")}{field("base_url", "API Base URL", "url")}
          {field("model", "模型 ID")}
          <label>API Key<input name="api_key" aria-label="API Key" type="password" autoComplete="off"
            value={key} onChange={e => setKey(e.target.value)}
            placeholder={connectionId ? "留空保留已有 Key" : "留空使用环境变量"} /></label>
        </div>
      </fieldset>
      <fieldset>
        <legend>用量与价格</legend>
        <div className="harness-grid">
          {field("context_window", "模型上下文上限（Token）", "number")}
          {field("max_output_tokens", "单次输出上限（Token）", "number")}
          {field("input_per_million_cny", "输入价格（元 / 百万 Token）", "number")}
          {field("output_per_million_cny", "输出价格（元 / 百万 Token）", "number")}
          {field("cache_read_per_million_cny", "缓存命中价（元 / 百万 Token，可选）", "number", false)}
        </div>
      </fieldset>
      <label className="harness-model-default">
        <input type="checkbox" checked={makeDefault} onChange={e => setMakeDefault(e.target.checked)} /> 用于此场景的新聊天
      </label>
      <button type="submit" disabled={busy}>保存并选择连接</button>
      <button type="button" disabled={busy || !connectionId} onClick={remove}>删除连接</button>
    </form>
    <p className="harness-status" role="status">{status}</p>
  </section>;
}
