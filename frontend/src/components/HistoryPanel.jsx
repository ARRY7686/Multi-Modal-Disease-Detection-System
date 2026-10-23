import { MODES, pct, prettyLabel } from '../lib/domain'

const time = (ts) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

export default function HistoryPanel({ entries, activeId, onSelect }) {
  return (
    <div>
      <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {entries.map((e) => {
          const r = e.results[e.primary]
          const top = r?.predictions?.[0]
          return (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => onSelect(e)}
                className={`flex w-full items-center gap-3 rounded-2xl border p-2.5 text-left transition-colors ${
                  e.id === activeId ? 'border-violet/60 bg-violet-soft' : 'border-line hover:border-line-strong'
                }`}
              >
                {e.thumb ? (
                  <img src={e.thumb} alt="" className="h-11 w-11 shrink-0 rounded-xl bg-black object-cover" />
                ) : (
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/[0.06] text-[10px] uppercase tracking-wider text-ink-muted">
                    text
                  </span>
                )}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-ink">{top ? prettyLabel(top.disease) : 'No result'}</span>
                  <span className="flex items-center gap-1.5 text-xs text-ink-muted">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: MODES[r?.mode]?.color }} />
                    {MODES[r?.mode]?.label} · {top ? pct(top.confidence, 0) : ''} · {time(e.ts)}
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
      <p className="mt-3 text-xs text-ink-muted">Kept in this browser only. Heatmaps are not saved.</p>
    </div>
  )
}
