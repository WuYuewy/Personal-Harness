import { useEffect, useRef, useState } from "react";

function loadCharts() {
  if (window.Chart) return Promise.resolve(window.Chart);
  if (!window.__personalHarnessCharts) window.__personalHarnessCharts = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/public/vendor/chart.umd.min.js";
    script.onload = () => resolve(window.Chart);
    script.onerror = () => { window.__personalHarnessCharts = null; reject(new Error("图表加载失败")); };
    document.head.appendChild(script);
  });
  return window.__personalHarnessCharts;
}

function ChartView({ type, items, label }) {
  const canvas = useRef(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let chart, cancelled = false;
    setFailed(false);
    loadCharts().then(Chart => {
      if (cancelled || !canvas.current) return;
      const color = getComputedStyle(canvas.current).color;
      chart = new Chart(canvas.current, {
        type,
        data: {
          labels: items.map(item => item.label),
          datasets: [{
            label, data: items.map(item => item.count),
            borderColor: "#8b83ff", backgroundColor: type === "radar" ? "#8b83ff40" : "#8b83ffaa",
            borderWidth: 2, pointRadius: 3, borderRadius: 4,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          indexAxis: type === "bar" ? "y" : "x",
          plugins: { legend: { display: false }, tooltip: { enabled: true } },
          scales: type === "radar" ? {
            r: { beginAtZero: true, suggestedMax: Math.max(1, ...items.map(i => i.count)),
              ticks: { precision: 0, color, backdropColor: "transparent" },
              pointLabels: { color, font: { size: 12 } },
              grid: { color: "#88888840" }, angleLines: { color: "#88888840" } },
          } : {
            x: { beginAtZero: true, ticks: { precision: 0, color }, grid: { color: "#88888825" } },
            y: { ticks: { color }, grid: { display: false } },
          },
        },
      });
    }).catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; chart?.destroy(); };
  }, [type, items, label]);
  return <section className="insight-chart" aria-label={label}>
    <h3>{label}</h3>
    {items.some(item => item.count > 0) ? <div className="insight-canvas"><canvas ref={canvas} role="img" aria-label={label} /></div> : <p>暂无数据。记录偏好或开启采集并提交日常任务后，这里会显示实际计数。</p>}
    {failed && <p role="alert">图表暂不可用，下面的计数仍可查看。</p>}
    <details><summary>查看计数</summary><ul>{items.map(item => <li key={item.label}>{item.label}：{item.count}</li>)}</ul></details>
  </section>;
}

function MemoryCard({ item, kinds, busy, mutate }) {
  const [text, setText] = useState(item.text);
  const [kind, setKind] = useState(item.kind);
  useEffect(() => { setText(item.text); setKind(item.kind); }, [item.text, item.kind]);
  return <form className="insight-memory" onSubmit={event => {
    event.preventDefault(); mutate({ operation: "save", id: item.id, memory: { kind, text } });
  }}>
    <label>偏好类别<select value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(kinds).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
    <label>偏好内容<textarea value={text} maxLength={240} minLength={2} required onChange={event => setText(event.target.value)} /></label>
    <p className="insight-meta">{item.status === "pending" ? "由你的明确表达提取，尚未用于适配" : "已确认 · 可用于相关任务"}{item.thread_id ? <> · <a href={`/thread/${encodeURIComponent(item.thread_id)}`}>查看来源聊天</a></> : " · 手动添加或来源已删除"}</p>
    <button type="submit" disabled={busy}>{item.status === "pending" ? "确认采用" : "保存修改"}</button>
    <button type="button" disabled={busy} onClick={() => {
      const operation = item.status === "pending" ? "dismiss" : "delete";
      if (operation === "dismiss" || window.confirm("删除此条已确认偏好吗？它将不再参与后续任务适配。")) {
        mutate({operation, id: item.id});
      }
    }}>{item.status === "pending" ? "忽略建议" : "删除偏好"}</button>
  </form>;
}

export default function UserInsights() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("workflow");
  const [text, setText] = useState("");
  const [tab, setTab] = useState("knowledge");
  const inFlight = useRef(false);
  async function mutate(payload) {
    if (inFlight.current) return false;
    inFlight.current = true;
    setBusy(true); setStatus("");
    try {
      const result = await callAction({name: "user_insights", payload});
      const response = result?.response ?? result;
      if (!response?.ok) throw new Error(response?.message || "操作失败，请刷新后重试。");
      setData(response.snapshot);
      setStatus(payload.operation === "refresh" ? "已刷新最新数据。" : "已保存。后续任务会采用最新设置。");
      return true;
    } catch (error) { setStatus(error.message || "连接失败，请稍后重试。"); return false; }
    finally { inFlight.current = false; setBusy(false); }
  }
  useEffect(() => { mutate({operation: "refresh"}); }, []);
  const confirmed = data?.memories.filter(memory => memory.status === "confirmed") || [];
  const pending = data?.memories.filter(memory => memory.status === "pending") || [];
  return <div className="harness-panel insight-panel">
    <h2>关于你 · 偏好与洞察</h2>
    <p>把长期习惯留给自己的助手。候选偏好先由你确认；已确认偏好只在相关任务中作为参考，当前任务和明确的助手设置优先。</p>
    <div className="insight-toolbar"><button type="button" disabled={busy} onClick={() => mutate({operation: "refresh"})}>刷新洞察</button><button type="button" disabled={!data || busy} onClick={() => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
      const link = document.createElement("a"); link.href = url; link.download = "my-preferences-and-insights.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}>导出我的数据</button></div>
    <p className="harness-status" role="status">{status}</p>
    {data && <>
      <div className="insight-options">
        <label><input type="checkbox" checked={data.options.capture} disabled={busy} onChange={event => {
          const enabled = event.target.checked;
          setData({...data, options: {...data.options, capture: enabled}});
          mutate({operation: "configure", name: "capture", enabled});
        }} />从后续消息提取偏好建议和使用统计</label>
        <label><input type="checkbox" checked={data.options.use_memories} disabled={busy} onChange={event => {
          const enabled = event.target.checked;
          setData({...data, options: {...data.options, use_memories: enabled}});
          mutate({operation: "configure", name: "use_memories", enabled});
        }} />在相关任务中使用已确认偏好</label>
        <p className="insight-capture-state" role="status">
          {data.options.capture
            ? "自动采集已开启：只分析本次及之后提交的普通消息；明确表达的长期习惯会先进入待确认建议。"
            : "自动采集已关闭：聊天不会更新偏好建议或使用画像。开启后只处理之后提交的普通消息，不会回溯已有聊天。"}
        </p>
        <p>本地分析，不额外调用模型，也不回溯旧聊天。只分析你发送的普通消息，不处理附件、工具结果或引用。额外保存统计、待确认建议和你确认的偏好，可随时关闭或清空；聊天记录仍按原有设置保存。</p>
      </div>
      <div role="tablist" aria-label="偏好与洞察视图" className="insight-tabs">
        <button type="button" role="tab" aria-selected={tab === "knowledge"} onClick={() => setTab("knowledge")}>偏好记忆 {confirmed.length} · 待确认 {pending.length}</button>
        <button type="button" role="tab" aria-selected={tab === "charts"} onClick={() => setTab("charts")}>使用画像</button>
      </div>
      {tab === "knowledge" ? <section role="tabpanel" aria-label="偏好记忆">
        <p className="insight-summary">已分析消息样本：{data.sample_count} · 已确认偏好：{confirmed.length} · 待确认建议：{pending.length}</p>
        <h3>添加一条长期偏好</h3>
        <form className="insight-memory" onSubmit={async event => {
          event.preventDefault(); if (await mutate({operation: "add", memory: {kind, text}})) setText("");
        }}>
          <label>新偏好类别<select value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(data.kinds).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
          <label>新偏好内容<textarea value={text} maxLength={240} minLength={2} required placeholder="例如：我通常用 Python 分析数据；我习惯每周五整理周报。" onChange={event => setText(event.target.value)} /></label>
          <button type="submit" disabled={busy}>添加并采用</button>
        </form>
        <h3>待确认建议 · {pending.length}</h3>
        {!pending.length && <p>开启采集后，像“我习惯每周五整理周报”这样的明确表达会出现在这里。任务中的临时要求不会自动成为长期偏好。</p>}
        {pending.map(item => <MemoryCard key={item.id} item={item} kinds={data.kinds} busy={busy} mutate={mutate} />)}
        <h3>已确认偏好 · {confirmed.length}</h3>
        {!confirmed.length && <p>你还没有记录长期偏好。先添加一条，或确认一条建议。</p>}
        {confirmed.map(item => <MemoryCard key={item.id} item={item} kinds={data.kinds} busy={busy} mutate={mutate} />)}
      </section> : <section role="tabpanel" aria-label="使用画像">
        <p>当前保留 {data.sample_count} 条消息样本，最多 2000 条。下面是已记录偏好与任务的实际频次，不是人格评分；一条任务可能有多个分类。</p>
        <div className="insight-charts">
          <ChartView type="radar" items={data.dimensions} label="已确认偏好维度（条）" />
          <ChartView type="bar" items={data.categories} label="日常任务分类（消息数）" />
          <ChartView type="bar" items={data.keywords} label="工作关键词（出现消息数）" />
          <ChartView type="bar" items={data.activity} label="最近有活动的日期（UTC）" />
        </div>
        <p>关键词由 jieba 本地提取，分类按公开词表匹配，可能有误；词频不自动变成长期记忆。</p>
      </section>}
      <div className="insight-reset"><button type="button" disabled={busy} onClick={() => {
        if (window.confirm("清空当前账号的所有偏好记忆、建议和使用统计，并关闭采集与使用？聊天记录及助手配置保留。")) mutate({operation: "clear"});
      }}>清空偏好与统计</button><p>删除来源聊天会移除该聊天统计和待确认建议；你已确认的偏好会保留，可单独删除。已发送给模型或出现在旧聊天中的内容无法撤回。</p></div>
    </>}
  </div>;
}
