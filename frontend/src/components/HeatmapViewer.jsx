import { useEffect, useState } from 'react'
import { prettyLabel } from '../lib/domain'
import { Close, Expand } from './Icons'

const VIEWS = [
  ['overlay', 'Overlay'],
  ['split', 'Side by side'],
]

function Stage({ original, heatmap, view, opacity, tall }) {
  const h = tall ? 'h-[78vh]' : 'h-72'
  if (view === 'split' || !original) {
    return (
      <div className={`grid gap-2 ${original ? 'grid-cols-2' : 'grid-cols-1'}`}>
        {original && (
          <figure className="overflow-hidden rounded-xl bg-black">
            <img src={original} alt="Original" className={`${h} w-full object-contain`} />
            <figcaption className="px-2 py-1 text-[11px] text-ink-muted">Original</figcaption>
          </figure>
        )}
        <figure className="overflow-hidden rounded-xl bg-black">
          <img src={heatmap} alt="Grad-CAM overlay" className={`${h} w-full object-contain`} />
          <figcaption className="px-2 py-1 text-[11px] text-ink-muted">Grad-CAM</figcaption>
        </figure>
      </div>
    )
  }
  return (
    <div className={`relative overflow-hidden rounded-xl bg-black ${h}`}>
      <img src={original} alt="Original" className="absolute inset-0 h-full w-full object-contain" />
      <img
        src={heatmap}
        alt="Grad-CAM overlay"
        className="absolute inset-0 h-full w-full object-contain"
        style={{ opacity }}
      />
    </div>
  )
}

export default function HeatmapViewer({ explanation, original }) {
  const [view, setView] = useState('overlay')
  const [opacity, setOpacity] = useState(0.85)
  const [full, setFull] = useState(false)

  useEffect(() => {
    if (!full) return
    const onKey = (e) => e.key === 'Escape' && setFull(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [full])

  if (!explanation?.heatmap_base64) return null
  const heatmap = `data:image/png;base64,${explanation.heatmap_base64}`

  const controls = (
    <div className="flex flex-wrap items-center gap-3">
      {original && (
        <div className="seg w-52">
          {VIEWS.map(([k, label]) => (
            <button
              key={k}
              type="button"
              onClick={() => setView(k)}
              className={`seg-btn ${view === k ? 'seg-btn-on' : ''}`}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {original && view === 'overlay' && (
        <label className="flex items-center gap-2 text-xs text-ink-2">
          Heatmap
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-28 accent-[#3987e5]"
          />
          <span className="num w-9 text-ink-muted">{Math.round(opacity * 100)}%</span>
        </label>
      )}
    </div>
  )

  return (
    <section className="card card-pad">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="card-title">Where the model looked</h2>
          <p className="text-xs text-ink-muted">
            Grad-CAM for <span className="text-ink-2">{prettyLabel(explanation.heatmap_class)}</span>
          </p>
        </div>
        <button type="button" onClick={() => setFull(true)} className="btn-ghost px-2.5 py-1.5 no-print" aria-label="Full screen">
          <Expand />
        </button>
      </div>
      <div className="mb-3 no-print">{controls}</div>
      <Stage original={original} heatmap={heatmap} view={view} opacity={opacity} />
      <p className="mt-3 text-[11px] leading-relaxed text-ink-muted">
        Warm regions pushed the score for the top class up. Grad-CAM shows what the network used, not where the
        disease is: a hot spot on a marker or the image border is a warning sign, not a finding.
      </p>

      {full && (
        <div className="fixed inset-0 z-50 flex flex-col bg-black/95 p-4 sm:p-8" role="dialog" aria-modal="true">
          <div className="mb-4 flex items-center justify-between gap-4">
            {controls}
            <button type="button" onClick={() => setFull(false)} className="btn-ghost px-2.5 py-1.5" aria-label="Close">
              <Close />
            </button>
          </div>
          <Stage original={original} heatmap={heatmap} view={view} opacity={opacity} tall />
        </div>
      )}
    </section>
  )
}
