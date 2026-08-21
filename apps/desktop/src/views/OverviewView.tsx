import type { BackendCapability, TaskSummary } from "../api";
import { Icon } from "../components/Icon";
import type { View } from "../lib/types";

export function OverviewView({
 setView, supportedCount, capabilityItems, runningTasks, recentPaths, setInputPath,
}: {
  setView: (view: View) => void;
  supportedCount: number;
  capabilityItems: BackendCapability["capabilities"];
  runningTasks: TaskSummary[];
  recentPaths: string[];
  setInputPath: (path: string) => void;
}) {
  return (

            <>
              <section className="welcome-panel">
                <div className="welcome-copy">
                  <span className="eyebrow"><Icon name="spark" size={14} /> NATIVE-FIRST DATA WORKSPACE</span>
                  <h1>让每一次转换<br /><em>都可解释、可恢复。</em></h1>
                  <p>从原始科学数据到高性能 Zarr。检查结构、配置处理流程，并在一个清晰的工作台中追踪每个结果。</p>
                  <div className="button-row">
                    <button className="primary-button large" type="button" onClick={() => setView("inspection")}><Icon name="upload" size={17} />开始检查数据</button>
                    <button className="quiet-button large" type="button" onClick={() => setView("tasks")}><Icon name="activity" size={17} />查看任务</button>
                  </div>
                </div>
                <div className="welcome-visual" aria-hidden="true">
                  <div className="visual-grid" />
                  <div className="visual-orbit orbit-one" />
                  <div className="visual-orbit orbit-two" />
                  <div className="visual-core"><Icon name="database" size={32} /></div>
                  <div className="visual-chip chip-top"><span />ZARR V3</div>
                  <div className="visual-chip chip-bottom"><Icon name="activity" size={13} /> READY TO FLOW</div>
                </div>
              </section>

              <div className="metric-grid">
                <article className="metric-card"><div className="metric-icon blue"><Icon name="layers" /></div><div><span>原生能力</span><strong>{supportedCount}<small> / {capabilityItems.length || "—"}</small></strong><p>已通过能力矩阵</p></div></article>
                <article className="metric-card"><div className="metric-icon violet"><Icon name="activity" /></div><div><span>活动任务</span><strong>{runningTasks.length}</strong><p>{runningTasks.length ? "正在执行" : "当前没有运行任务"}</p></div></article>
                <article className="metric-card"><div className="metric-icon green"><Icon name="folder" /></div><div><span>最近路径</span><strong>{recentPaths.length}</strong><p>本地工作区记录</p></div></article>
                <article className="metric-card accent"><div className="metric-icon orange"><Icon name="spark" /></div><div><span>执行策略</span><strong>Auto</strong><p>按能力自动路由</p></div></article>
              </div>

              <div className="dashboard-grid">
                <article className="surface recent-card">
                  <div className="section-heading"><div><span className="section-kicker">QUICK ACCESS</span><h2>最近使用</h2></div><button className="link-button" type="button" onClick={() => setView("settings")}>管理路径 <Icon name="arrow" size={14} /></button></div>
                  {recentPaths.length ? <div className="recent-list">{recentPaths.slice(0, 4).map((path) => <button className="recent-item" type="button" key={path} onClick={() => { setInputPath(path); setView("inspection"); }}><span className="recent-icon"><Icon name="folder" size={16} /></span><span><strong>{path.split(/[\\/]/).pop() || path}</strong><small>{path}</small></span><Icon name="chevron" size={15} /></button>)}</div> : <div className="empty-inline"><Icon name="folder" size={22} /><p>还没有最近路径<br /><button type="button" onClick={() => setView("inspection")}>选择一个数据目录开始</button></p></div>}
                </article>
                <article className="surface route-card">
                  <div className="section-heading"><div><span className="section-kicker">EXECUTION ROUTE</span><h2>执行路线</h2></div><button className="link-button" type="button" onClick={() => setView("settings")}>查看设置 <Icon name="arrow" size={14} /></button></div>
                  <div className="route-list"><div className="route-item"><span className="route-number">01</span><div><strong>检查输入结构</strong><small>识别变量、维度和时间规则</small></div><span className="route-status ready">READY</span></div><div className="route-item"><span className="route-number">02</span><div><strong>按能力编排</strong><small>原生路径优先，兼容路径兜底</small></div><span className="route-status ready">AUTO</span></div><div className="route-item"><span className="route-number">03</span><div><strong>校验并发布</strong><small>staging 完成后原子发布结果</small></div><span className="route-status wait">SAFE</span></div></div>
                </article>
              </div>
            </>

  );
}
