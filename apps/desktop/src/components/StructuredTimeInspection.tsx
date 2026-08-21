import type { TimeDimensionSummary, TimeInspection } from "../api";
import { fieldValuesPreview, timeValuesPreview } from "../lib/format";

export function StructuredTimeInspection({ inspection }: { inspection: TimeInspection }) {
  const dimension: TimeDimensionSummary = inspection.time_dimension;
  const attributes = Object.entries(dimension.attrs).filter(([, value]) => value !== null && value !== "").slice(0, 4);
  return (
    <div className="structured-time-report">
      <div className="inspection-report-grid">
        <div><span>文件数量</span><strong>{inspection.files.length.toLocaleString()}</strong></div>
        <div><span>读取引擎</span><strong>{inspection.engine}</strong></div>
        <div><span>数据维度</span><strong>{inspection.dimensions.length || "—"}</strong><small>{inspection.dimensions.join(" · ") || "未识别"}</small></div>
        <div><span>坐标数量</span><strong>{inspection.coordinates.length || "—"}</strong><small>{inspection.coordinates.join(" · ") || "未识别"}</small></div>
      </div>
      <section className="report-section">
        <div className="report-section-heading"><strong>文件名时间字段</strong><span>{inspection.filename_fields.length} 个数字字段</span></div>
        {inspection.filename_fields.length ? (
          <div className="filename-field-list">
            {inspection.filename_fields.map((field) => (
              <div className="filename-field-card" key={field.index}>
                <div className="filename-field-head"><strong>字段 #{field.index}</strong><span className={field.changed ? "field-status changed" : "field-status"}>{field.changed ? "跨文件变化" : "稳定/未验证"}</span></div>
                <div className="filename-field-meta"><span>样例 <b>{field.sample}</b></span><span>位置 {field.start}–{field.start + field.length - 1}</span><span>长度 {field.length}</span></div>
                <small>{fieldValuesPreview(field)}</small>
              </div>
            ))}
          </div>
        ) : <p className="report-empty">未发现可用于时间解析的文件名数字字段。</p>}
      </section>
      <section className="report-section">
        <div className="report-section-heading"><strong>数据内时间维度</strong><span className={dimension.exists ? "report-status good" : "report-status warning"}>{dimension.exists ? "已发现" : "未发现"}</span></div>
        {dimension.exists ? (
          <div className="report-detail-grid">
            <div><span>维度名称</span><strong>{dimension.name || "未命名"}</strong></div>
            <div><span>格式</span><strong>{dimension.format_label || "未识别"}</strong></div>
            <div className="wide"><span>解码样例</span><strong>{timeValuesPreview(dimension.decoded_values)}</strong></div>
            {attributes.map(([key, value]) => <div key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}
          </div>
        ) : <p className="report-empty">首个文件没有可识别的 time 维度，将依赖文件名规则生成时间轴。</p>}
      </section>
      <section className="report-section">
        <div className="report-section-heading"><strong>可用时间规则</strong><span>{inspection.options.length} 个候选</span></div>
        {inspection.options.length ? <div className="rule-option-list">{inspection.options.map((option) => <span key={`${option.ref.source}:${option.ref.component}:${option.ref.index}`}>{option.label}</span>)}</div> : <p className="report-empty">暂无可用候选，请检查输入文件的命名或时间维度。</p>}
      </section>
      <details className="raw-inspection-report"><summary>查看原始检查日志</summary><pre>{inspection.report}</pre></details>
    </div>
  );
}