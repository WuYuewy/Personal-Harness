export default function RunReceipt() {
  const { offline, currency, cost, receipt, rehearsal, tokens, profile, availableTools,
    cacheReadTokens, cacheSavings,
    historyReadBytesOmitted, historyReadsReferenced,
    pagedReadResults, readResultBytesOmitted,
    toolCalls, preferences, runId } = props;
  const price = currency === "CNY" ? `¥${cost}` : `${currency} ${cost}`;
  const summary = offline ? "离线演练 · 无 API 费用" : cost === null ? "费用待核对" : `费用 ${price}`;
  return <details className="harness-receipt">
    <summary>
      <span>{summary}</span>
      {!offline && tokens > 0 && <span> · {tokens.toLocaleString()} Token</span>}
      <span className="harness-receipt-label">执行详情</span>
    </summary>
    <div className="harness-receipt-details">
      <p>场景 {profile} · 启用 {availableTools} 个工具 · 调用 {toolCalls} 次</p>
      {preferences > 0 && <p>本轮参考 {preferences} 条已确认的个人偏好，可在工具箱中管理。</p>}
      {historyReadsReferenced > 0 && <p>Harness 已在请求中将 {historyReadsReferenced} 份较早的文件正文替换为来源引用，
        累计减少 {(historyReadBytesOmitted / 1024).toFixed(1)} KiB 重复发送内容。
        完整历史仍保存在本地；此为请求正文精简量，不代表已验证的费用节省。</p>}
      {pagedReadResults > 0 && <p>已对 {pagedReadResults} 份大段读取结果分页，暂未返回
        {(readResultBytesOmitted / 1024).toFixed(1)} KiB 正文；模型可搜索或续读原文件。
        这是工具返回量的变化，不代表任务已完成或已节省相应费用。</p>}
      {cacheReadTokens > 0 && <p>供应商报告 {cacheReadTokens.toLocaleString()} 个缓存读取输入 Token，
        {cacheSavings == null ? "价差无法完整计算（缺少价格或计量信息）。" :
          `按本连接配置价格估算，缓存折扣约 ¥${cacheSavings}；此项属于模型供应商计价，不属于 Personal Harness 优化，且未核对平台账单。`}</p>}
      <ul>{receipt.split("\n").map(line => <li key={line}>{line}</li>)}</ul>
      <p>任务编号 <code>{runId}</code></p>
      {rehearsal && <pre>{rehearsal}</pre>}
    </div>
  </details>;
}
