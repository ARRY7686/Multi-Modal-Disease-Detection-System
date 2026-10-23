import { MODES, pct, prettyLabel } from '../lib/domain'

const time = (ts) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

export default function HistoryPanel({ entries, activeId, onSelect, onClear }) {
  if (entries.length === 0) return null
  return (
    <section className="card card-pad no-print">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="card-title">Recent analyses</h2>
        <button type="button" onClick={onClear} className="text-xs text-ink-muted hover:text-ink">
          Clear
        </button>
      </div>
      <ul className="space-y-1">
        {entries.map((e) => {
          const r = e.results[e.primary]
          const top = r?.predictions?.[0]
          return (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => onSelect(e)}
                className={`flex w-full items-center gap-3 rounded-xl p-2 text-left transition-colors ${
                  e.id === activeId ? 'bg-surface-3' : 'hover:bg-surface-2'
                }`}
              >
                {e.thumb ? (
                  <img src={e.thumb} alt="" className="h-10 w-10 shrink-0 rounded-lg bg-black object-cover" />
                ) : (
                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-surface-3 text-[10px] text-ink-muted">
                    text
                  </span>
                )}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-ink">{top ? prettyLabel(top.disease) : 'No result'}</span>
                  <span className="flex items-center gap-1.5 text-xs text-ink-muted">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: MODES[r?.mode]?.color }} />
                    {MODES[r?.mode]?.label} · {top ? pct(top.confidence, 0) : ''}
                  </span>
                </span>
                <span className="num text-[11px] text-ink-muted">{time(e.ts)}</span>
              </button>
            </li>
          )
        })}
      </ul>
      <p className="mt-2 text-[11px] text-ink-muted">Kept in this browser only. Heatmaps are not saved.</p>
    </section>
  )
}
