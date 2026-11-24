import { useEffect, useRef, useState } from "react";

const tools = [
  ["/models", "模型连接", "管理 API 与模型"],
  ["/scenes", "个人场景", "保存和管理常用聊天设置"],
  ["/files", "工作区", "浏览任务文件"],
  ["/find", "文件搜索", "本地执行 · 无 API 费用"],
  ["/lookup", "CSV 查找", "按列精确查找 · 无 API 费用"],
  ["/sum", "CSV 汇总", "本地计算合计与平均 · 无 API 费用"],
  ["/actions", "提取待办", "按行首标记提取 · 保留来源行号 · 无 API 费用"],
  ["/places", "地点筛选", "本地执行 · 无 API 费用"],
  ["/reports", "个人任务方案", "保存可复用的文件简报"],
  ["/insights", "偏好与洞察", "管理习惯与使用画像"],
  ["/budget", "预算", "查看可用额度"],
  ["/usage", "本聊天费用", "查看用量与费用记录"],
  ["/help", "使用指南", "查看快捷命令"],
];

export default function WorkspaceWelcome() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [local, setLocal] = useState("");
  const [query, setQuery] = useState("");
  const [lookup, setLookup] = useState({ path: "", column: "", value: "" });
  const [summary, setSummary] = useState({ path: "", numeric_column: "", filter_column: "", filter_value: "" });
  const [actions, setActions] = useState({ path: "", prefix: "TODO:" });
  const [shortcuts, setShortcuts] = useState(props.initialShortcuts || []);
  const [shortcutName, setShortcutName] = useState("");
  const [editing, setEditing] = useState(null);
  const [notice, setNotice] = useState("");
  const pending = useRef(false);

  useEffect(() => {
    if (!props.menu) return;
    let mounted = true;
    callAction({ name: "local_shortcut", payload: { operation: "list" } })
      .then(result => {
      const data = result?.response ?? result;
      if (mounted && data?.ok) setShortcuts(data.items || []);
        else if (mounted && props.menu) setError(data?.message || "无法读取快捷查询。");
      })
      .catch(() => { if (mounted && props.menu) setError("无法读取快捷查询。"); });
    return () => { mounted = false; };
  }, []);

  async function changeShortcut(operation, id, shortcut) {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await callAction({
        name: "local_shortcut", payload: { operation, id, shortcut },
      });
      const data = result?.response ?? result;
      if (!data?.ok) {
        setError(data?.message || "快捷查询操作失败。");
        return;
      }
      if (data.items) setShortcuts(data.items);
      if (operation === "save" || operation === "delete") {
        if (operation === "save" || id === editing) {
          setEditing(null);
          setShortcutName("");
        }
        setNotice(operation === "save" ? "快捷查询已保存。" : "快捷查询已删除。");
      }
    } catch {
      setError("快捷查询操作失败，请重试。");
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }

  function editShortcut(item) {
    setLocal(item.command);
    setEditing(item.id);
    setShortcutName(item.name);
    setNotice("");
    setError("");
    if (item.command === "/lookup") {
      const [path, column, value] = item.args;
      setLookup({ path, column, value });
    } else if (item.command === "/sum") {
      const [path, numeric_column, filter_column, filter_value] = item.args;
      setSummary({ path, numeric_column, filter_column, filter_value });
    } else if (item.command === "/actions") {
      const [path, prefix = "TODO:"] = item.args;
      setActions({ path, prefix });
    } else {
      setQuery(item.args[0]);
    }
  }

  const args = local === "/lookup" ? [lookup.path.trim(), lookup.column, lookup.value]
    : local === "/sum" ? [summary.path.trim(), summary.numeric_column, summary.filter_column, summary.filter_value]
    : local === "/actions" ? [actions.path.trim(), actions.prefix]
    : [query.trim()];
  let validQuery = args.every(value => value.trim());
  if (local === "/places") validQuery = true;
  if (local === "/sum") validQuery = args[0] && args[1] && Boolean(args[2]) === Boolean(args[3]);
  const canSave = local && shortcutName.trim() && validQuery;

  async function open(command, value, fields = {}) {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      await callAction({ name: "workspace_tool", payload: { command, query: value, ...fields } });
    } catch {
      setError("打开失败，请重试。");
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }

  function choose(command) {
    if (["/find", "/places", "/lookup", "/sum", "/actions"].includes(command)) {
      setQuery("");
      setSummary({ path: "", numeric_column: "", filter_column: "", filter_value: "" });
      setActions({ path: "", prefix: "TODO:" });
      setEditing(null);
      setShortcutName("");
      setLocal(command);
    } else {
      open(command);
    }
  }

  const shortcutList = shortcuts.length > 0 && <section className="harness-shortcuts" aria-label="快捷查询">
    <h3>快捷查询</h3>
    <ul>{shortcuts.map(item => <li key={item.id}>
      <button type="button" disabled={busy} onClick={() => changeShortcut("run", item.id)}>
        {item.name}
      </button>
      {props.menu && <button type="button" disabled={busy}
        aria-label={`编辑 ${item.name}`} onClick={() => editShortcut(item)}>编辑</button>}
      {props.menu && <button type="button" disabled={busy}
        aria-label={`删除 ${item.name}`} onClick={() => changeShortcut("delete", item.id)}>删除</button>}
    </li>)}</ul>
  </section>;

  if (props.menu) {
    return <section className="harness-panel harness-toolbox" aria-label="工具箱">
      <h2>工具箱</h2>
      <div className="harness-tool-grid">
        {tools.map(([command, label, description]) => <button key={command} type="button"
          disabled={busy} onClick={() => choose(command)}>
          <strong>{label}</strong><span>{description}</span>
        </button>)}
      </div>
      {shortcutList}
      {local === "/lookup" && <form className="harness-local-form" onSubmit={(event) => {
        event.preventDefault();
        open(local, undefined, { ...lookup, path: lookup.path.trim() });
      }}>
        {[["path", "工作区 CSV 路径", "expenses.csv"], ["column", "列名", "invoice_id"], ["value", "精确值", "INV-0173"]].map(([key, label, placeholder]) =>
          <label key={key}>{label}<input type="text" value={lookup[key]} placeholder={placeholder}
            maxLength={240} disabled={busy} required onChange={event => setLookup({ ...lookup, [key]: event.target.value })} /></label>)}
        <button type="submit" disabled={busy || !lookup.path.trim() || !lookup.column || !lookup.value}>查找</button>
      </form>}
      {local === "/sum" && <form className="harness-local-form" onSubmit={(event) => {
        event.preventDefault();
        open(local, undefined, summary);
      }}>
        {[["path", "工作区 CSV 路径", "expenses.csv"], ["numeric_column", "数字列名", "金额元"],
          ["filter_column", "筛选列（选填）", "状态"], ["filter_value", "筛选值（选填）", "待报销"]].map(([key, label, placeholder]) =>
          <label key={key}>{label}<input type="text" value={summary[key]} placeholder={placeholder}
            maxLength={240} disabled={busy} onChange={event => setSummary({ ...summary, [key]: event.target.value })} /></label>)}
        <button type="submit" disabled={busy || !summary.path.trim() || !summary.numeric_column.trim() ||
          Boolean(summary.filter_column) !== Boolean(summary.filter_value)}>汇总</button>
      </form>}
      {local === "/actions" && <form className="harness-local-form" onSubmit={(event) => {
        event.preventDefault();
        open(local, undefined, { ...actions, path: actions.path.trim() });
      }}>
        {[["path", "工作区文本文件路径", "activity-log.md"], ["prefix", "行首标记", "TODO:"]].map(([key, label, placeholder]) =>
          <label key={key}>{label}<input type="text" value={actions[key]} placeholder={placeholder}
            maxLength={key === "prefix" ? 80 : 240} disabled={busy} required
            onChange={event => setActions({ ...actions, [key]: event.target.value })} /></label>)}
        <button type="submit" disabled={busy || !actions.path.trim() || !actions.prefix.trim()}>提取</button>
      </form>}
      {local && !["/lookup", "/sum", "/actions"].includes(local) && <form className="harness-local-form" onSubmit={(event) => {
        event.preventDefault();
        if (local === "/places" || query.trim()) open(local, query.trim());
      }}>
        <label>{local === "/find" ? "搜索关键词" : "地点类别（选填）"}
          <input type="text" value={query} maxLength={200} disabled={busy}
            required={local === "/find"} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <button type="submit" disabled={busy || (local === "/find" && !query.trim())}>
          {local === "/find" ? "搜索" : "筛选"}
        </button>
      </form>}
      {local && <div className="harness-local-form">
        <label>快捷查询名称
          <input type="text" value={shortcutName} maxLength={48} disabled={busy}
            onChange={event => setShortcutName(event.target.value)} />
        </label>
        <button type="button" disabled={busy || !canSave}
          onClick={() => changeShortcut("save", editing, { name: shortcutName.trim(), command: local, args })}>
          {editing ? "更新快捷查询" : "保存快捷查询"}
        </button>
        {editing && <button type="button" disabled={busy}
          onClick={() => { setEditing(null); setShortcutName(""); setLocal(""); }}>取消编辑</button>}
      </div>}
      <p className="harness-status" role="status">{error || notice}</p>
    </section>;
  }

  return <section className="harness-welcome" aria-label="新聊天">
    <div className="harness-welcome-title">
      <img src="/favicon" width="42" height="42" alt="" />
      <h1>今天想做什么？</h1>
    </div>
    {shortcutList}
    <p>在下方描述任务，也可以附上需要处理的资料。</p>
    <div className="harness-welcome-details">
      <span className="harness-mode" data-online={!props.offline}>
        {props.offline ? "离线演练" : "在线模型"}
      </span>
      <span className="harness-model-name" title={props.model}>{props.model}</span>
      {props.offline && <button type="button" disabled={busy} onClick={() => open("/models")}>
        配置模型
      </button>}
      <button type="button" disabled={busy} onClick={() => open("/tools")}>打开工具箱</button>
    </div>
    <p className="harness-status" role="status">{error}</p>
  </section>;
}
