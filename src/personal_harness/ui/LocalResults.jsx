export default function LocalResults() {
  const data = props;
  if (data.actions_result) return <section className="harness-panel harness-local-results" aria-label="待办提取结果">
    <h2>待办提取结果</h2>
    <p>本地执行 · 0 次模型调用 · ¥0 API 费用</p>
    <p><code>{data.path}</code> · 标记 <code>{data.prefix}</code> · 共 {data.match_count} 项
      {data.truncated && <> · 显示 {data.displayed_count} 项</>}
    </p>
    {data.match_count === 0 && <p>没有找到以该标记开头的行。</p>}
    {data.actions.length > 0 &&
      <ol>{data.actions.map((item, index) =>
        <li key={index}><code>{item.path}:{item.line}</code><span>{item.text}</span></li>)}</ol>}
    {data.truncated && <p>结果达到本地展示上限，另有 {data.omitted_count} 项未显示。请缩小来源文件或使用更具体的标记。</p>}
    <details><summary>来源校验</summary><code>SHA-256 {data.sha256}</code></details>
  </section>;
  if (data.summary) return <section className="harness-panel harness-local-results" aria-label="CSV 汇总结果">
    <h2>CSV 汇总结果</h2>
    <p>本地执行 · 0 次模型调用 · ¥0 API 费用</p>
    <p><code>{data.path}</code> · {data.numeric_column}</p>
    {data.filter_column && <p>筛选：{data.filter_column} = {data.filter_value}</p>}
    <dl className="harness-summary-stats">
      <div><dt>匹配条数</dt><dd>{data.matched_count}</dd></div>
      <div><dt>合计</dt><dd>{data.total}</dd></div>
      <div><dt>平均值</dt><dd>{data.average}</dd></div>
    </dl>
  </section>;
  if (data.csv) return <section className="harness-panel harness-local-results" aria-label="CSV 查找结果">
    <h2>CSV 查找结果</h2>
    <p>本地执行 · 0 次模型调用 · ¥0 API 费用</p>
    <p><code>{data.path}</code> · {data.match_count} 项匹配</p>
    {data.match_count === 0 && <p>没有找到精确匹配。</p>}
    <ul>{data.rows.map(row => <li key={row.start_line}>
      <code>第 {row.start_line}{row.end_line !== row.start_line && `–${row.end_line}`} 行</code>
      <dl className="harness-csv-row">{row.values.map((value, index) => <div key={index}>
        <dt>{data.headers[index]}</dt><dd>{value}</dd>
      </div>)}</dl>
    </li>)}</ul>
    {data.truncated && <p>受结果上限限制，显示 {data.rows.length} / {data.match_count} 项完整记录。</p>}
  </section>;
  const rows = data.search ? data.matches : data.places;
  return <section className="harness-panel harness-local-results">
    <h2>{data.search ? "工作区搜索结果" : "本地地点筛选结果"}</h2>
    <p>本地执行 · 0 次模型调用 · ¥0 API 费用</p>
    {!data.search && <p>
      {data.source} · 采集于 {data.fetched_at}<br />
      {data.limits.max_distance_m} 米内（直线距离）
      {data.limits.max_price_cny !== null && ` · ¥${data.limits.max_price_cny} 以内`}<br />
      {data.timestamp_in_future ? "来源时间晚于当前时间。" : data.stale ? "来源已过期。" : ""}
      营业状态未实时核验。
    </p>}
    {!data.search && data.unknown_price_excluded > 0 && <p>已排除 {data.unknown_price_excluded} 项价格未知的地点。</p>}
    {rows.length === 0 ? <p>{!data.search && !data.city_matches ? "来源城市与当前配置不同。" : "没有找到符合条件的结果。"}</p> :
      <ul>{rows.map((item, index) => data.search ?
        <li key={index}><code>{item.path}:{item.line}</code><span>{item.text}</span></li> :
        <li key={item.id}><strong>{item.name}</strong>
          <span>{item.category} · {item.distance_m} 米 · {item.price_cny == null ? "价格未知" : `¥${item.price_cny}`}</span>
          {item.address && <span>{item.address}</span>}
        </li>)}</ul>}
    {data.search && <p>扫描 {data.files_scanned} 个文件；跳过 {data.skipped} 个读取项。</p>}
    {data.truncated && <p>{data.search ? "扫描或片段受到配置上限限制，结果可能不完整。" : "符合条件的地点超过展示上限。"}</p>}
  </section>;
}
