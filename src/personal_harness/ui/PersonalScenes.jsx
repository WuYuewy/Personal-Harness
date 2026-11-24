import { useEffect, useState } from "react";

export default function PersonalScenes() {
  const [items, setItems] = useState([]);
  const [current, setCurrent] = useState(null);
  const [form, setForm] = useState({ operation: "create", name: "" });
  const [busy, setBusy] = useState(true);
  const [notice, setNotice] = useState("");
  const [refresh, setRefresh] = useState(false);

  async function act(operation, fields = {}) {
    setBusy(true);
    setNotice("");
    try {
      const response = await callAction({ name: "personal_scene", payload: { operation, ...fields } });
      const data = response?.response ?? response;
      if (!data?.ok) {
        setNotice(data?.message || "场景操作失败。");
        return;
      }
      setItems(data.items);
      setCurrent(data.current);
      if (operation === "list") return;
      setForm({ operation: "create", name: "" });
      setNotice(operation === "save_current" ? "已更新此场景的新聊天默认设置。" : "已保存场景变更。");
      if (operation !== "save_current") setRefresh(true);
    } catch {
      setNotice("无法读取或保存场景，请重试。");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { act("list"); }, []);

  const selected = items.find(item => item.id === current);
  const names = { create: "从当前聊天创建场景", rename: "重命名场景", copy: "复制场景" };
  return <section className="harness-panel harness-scenes" aria-label="个人场景">
    <h2>个人场景</h2>
    <p>当前场景：{selected?.name || current || "默认场景"}</p>
    {selected && !selected.archived && <button type="button" disabled={busy}
      onClick={() => act("save_current")}>用当前聊天设置更新此场景</button>}
    {items.length > 0 && <ul>
      {items.map(item => <li key={item.id}>
        <strong>{item.name}</strong>
        <span>{item.archived ? "已归档" : item.base_profile}</span>
        <div className="harness-scene-actions">
          <button type="button" disabled={busy} aria-label={"改名 " + item.name}
            onClick={() => setForm({ operation: "rename", id: item.id, name: item.name })}>改名</button>
          <button type="button" disabled={busy} aria-label={"复制 " + item.name}
            onClick={() => setForm({ operation: "copy", id: item.id, name: "" })}>复制</button>
          <button type="button" disabled={busy} aria-label={(item.archived ? "恢复 " : "归档 ") + item.name}
            onClick={() => act(item.archived ? "restore" : "archive", { id: item.id })}>
            {item.archived ? "恢复" : "归档"}
          </button>
        </div>
      </li>)}
    </ul>}
    <form className="harness-local-form" onSubmit={event => {
      event.preventDefault();
      if (!busy && form.name.trim()) act(form.operation, { id: form.id, name: form.name.trim() });
    }}>
      <label>场景名称
        <input type="text" maxLength={48} required value={form.name} disabled={busy}
          onChange={event => setForm({ ...form, name: event.target.value })} />
      </label>
      <button type="submit" disabled={busy || !form.name.trim()}>{names[form.operation]}</button>
      {form.operation !== "create" && <button type="button" disabled={busy}
        onClick={() => setForm({ operation: "create", name: "" })}>取消</button>}
    </form>
    <p role="status" className="harness-status">{notice}</p>
    {refresh && <button type="button" disabled={busy} onClick={() => window.location.reload()}>
      刷新可选场景
    </button>}
  </section>;
}
