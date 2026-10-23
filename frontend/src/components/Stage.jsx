import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { findHotspot } from '../lib/hotspot'
import { MODALITIES, SOURCE_LABELS, pct, prettyLabel } from '../lib/domain'
import { Alert, Close, Expand, Upload } from './Icons'

const MAX_MB = 10

// Callout slots around the visual (md and up); stacked below it on phones.
const SLOT = {
  tl: 'md:absolute md:left-0 md:top-[7%]',
  tr: 'md:absolute md:right-0 md:top-[4%]',
  bl: 'md:absolute md:left-[2%] md:bottom-[9%]',
  br: 'md:absolute md:right-[1%] md:bottom-[5%]',
}

function Callout({ slot, cardRef, eyebrow, title, sub, accent, children }) {
  return (
    <div ref={cardRef} className={`glass-card w-full p-4 md:w-60 ${SLOT[slot]}`}>
      {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
      <div className="flex items-start gap-2.5">
        <span className="tri mt-1.5 rotate-90" style={{ borderBottomColor: accent || '#a99ff0' }} />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] text-ink">{title}</div>
          {sub && <div className="mt-0.5 text-xs text-ink-muted">{sub}</div>}
          {children}
        </div>
      </div>
    </div>
  )
}

function FindingCallout({ p, slot, cardRef, eyebrow, override }) {
  return (
    <Callout
      slot={slot}
      cardRef={cardRef}
      eyebrow={eyebrow}
      title={prettyLabel(p.disease)}
      accent={p.positive ? '#ec835a' : '#a99ff0'}
      sub={p.positive ? 'Above its decision threshold' : 'Below its decision threshold'}
    >
      <div className="mt-2.5 flex items-center gap-2">
        <div className="relative h-1 flex-1 rounded-full bg-white/10">
          <div className="h-1 rounded-full bg-violet-strong" style={{ width: `${Math.max(p.confidence * 100, 1)}%` }} />
          <div className="absolute -top-1 h-3 w-px bg-ink-2" style={{ left: `${p.threshold * 100}%` }} />
        </div>
        <span className="num text-sm text-ink">{pct(p.confidence)}</span>
      </div>
      {override && (
        <div className="mt-3 flex gap-1.5 border-t border-line pt-2.5 text-[11px] leading-snug text-status-warning">
          <Alert className="mt-px h-3 w-3 shrink-0" />
          <span>
            Image only says {prettyLabel(override.image.disease)} ({pct(override.image.confidence)})
          </span>
        </div>
      )}
    </Callout>
  )
}

// Thin leader lines from each callout to a point on the visual, drawn in
// stage pixels after layout (and again whenever the stage resizes).
function useLeaders(stageRef, targetRef, cards, deps) {
  const [lines, setLines] = useState([])
  const measure = useCallback(() => {
    const stage = stageRef.current
    const target = targetRef.current
    if (!stage || !target || window.innerWidth < 768) return setLines([])
    const s = stage.getBoundingClientRect()
    const t = target.getBoundingClientRect()
    setLines(
      cards
        .filter((c) => c.ref.current)
        .map((c) => {
          const r = c.ref.current.getBoundingClientRect()
          const leftSide = r.left + r.width / 2 < t.left + t.width / 2
          return {
            key: c.key,
            x1: (leftSide ? r.right : r.left) - s.left,
            y1: r.top + Math.min(r.height / 2, 30) - s.top,
            x2: t.left + c.fx * t.width - s.left,
            y2: t.top + c.fy * t.height - s.top,
          }
        }),
    )
  }, [stageRef, targetRef, cards])

  useLayoutEffect(() => {
    measure()
    const ro = new ResizeObserver(measure)
    if (stageRef.current) ro.observe(stageRef.current)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return [lines, measure]
}

export default function Stage({ preview, fileName, onImage, result, override, loading }) {
  const stageRef = useRef(null)
  const targetRef = useRef(null)
  const inputRef = useRef(null)
  const refs = { tl: useRef(null), tr: useRef(null), bl: useRef(null), br: useRef(null) }
  const [ratio, setRatio] = useState(1)
  const [view, setView] = useState('overlay')
  const [opacity, setOpacity] = useState(0.85)
  const [hotspot, setHotspot] = useState(null)
  const [full, setFull] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)

  const heatmap = result?.explanation?.heatmap_base64 ? `data:image/png;base64,${result.explanation.heatmap_base64}` : null
  const preds = result?.predictions || []

  const accept = useCallback(
    (file) => {
      if (!file) return
      if (!file.type.startsWith('image/')) return setError('That file is not an image.')
      if (file.size > MAX_MB * 1024 * 1024) return setError(`Images up to ${MAX_MB} MB.`)
      setError(null)
      onImage(file)
    },
    [onImage],
  )

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
      if (item) accept(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [accept])

  useEffect(() => {
    setHotspot(null)
    if (heatmap) findHotspot(heatmap).then(setHotspot)
  }, [heatmap])

  useEffect(() => {
    if (!full) return
    const onKey = (e) => e.key === 'Escape' && setFull(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [full])

  // Which callouts show, and where each one's leader line lands (0..1 of the visual)
  let cards = []
  if (!preview) {
    cards = [
      { key: 'tl', fx: 0.12, fy: 0.3 },
      { key: 'tr', fx: 0.88, fy: 0.25 },
      { key: 'bl', fx: 0.2, fy: 0.85 },
    ]
  } else if (result) {
    cards = [
      { key: 'tl', fx: 0, fy: 0.08 },
      ...(preds[0] ? [{ key: 'tr', fx: hotspot?.x ?? 0.5, fy: hotspot?.y ?? 0.5 }] : []),
      ...(preds[1] ? [{ key: 'bl', fx: 0, fy: 0.78 }] : []),
      ...(preds[2] ? [{ key: 'br', fx: 1, fy: 0.9 }] : []),
    ]
  } else {
    cards = [{ key: 'tl', fx: 0, fy: 0.1 }]
  }
  cards = cards.map((c) => ({ ...c, ref: refs[c.key] }))
  const [lines, remeasure] = useLeaders(stageRef, targetRef, cards, [preview, result, hotspot, ratio])

  const frameStyle = { aspectRatio: `${ratio}` }
  const visual = (tall = false) => (
    <div
      className={`relative overflow-hidden rounded-2xl border border-line-strong bg-black shadow-[0_0_120px_rgba(144,133,233,0.28)] ${
        tall ? 'max-h-[80vh]' : ''
      }`}
      style={frameStyle}
    >
      <img
        src={preview}
        alt="Uploaded scan"
        className="absolute inset-0 h-full w-full object-contain"
        onLoad={(e) => {
          setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight || 1)
          requestAnimationFrame(remeasure)
        }}
      />
      {heatmap && view === 'overlay' && (
        <img src={heatmap} alt="Grad-CAM overlay" className="absolute inset-0 h-full w-full object-contain" style={{ opacity }} />
      )}
      {heatmap && hotspot && view === 'overlay' && (
        <span
          className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${hotspot.x * 100}%`, top: `${hotspot.y * 100}%` }}
        >
          <span className="absolute inset-0 animate-ping-soft rounded-full bg-white/70" />
          <span className="absolute inset-0 rounded-full border-2 border-white bg-violet-strong" />
        </span>
      )}
      {loading && (
        <span className="absolute inset-x-0 h-16 animate-scan bg-gradient-to-b from-transparent via-violet/40 to-transparent" />
      )}
    </div>
  )

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        accept(e.dataTransfer.files?.[0])
      }}
    >
      <div ref={stageRef} className="relative flex min-h-[420px] flex-col items-center gap-4 md:block md:min-h-[600px]">
        {/* Decoration: glow and slow orbit rings */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center" aria-hidden="true">
          <div className="absolute h-[380px] w-[380px] rounded-full bg-violet-strong/25 blur-[90px]" />
          <div className="absolute ml-40 mt-32 h-[260px] w-[260px] rounded-full bg-magenta/15 blur-[90px]" />
          <svg viewBox="0 0 600 600" className="absolute h-[600px] w-[600px] animate-spin-slow opacity-60">
            <circle cx="300" cy="300" r="290" fill="none" stroke="rgba(169,159,240,0.18)" strokeDasharray="2 10" />
            <circle cx="300" cy="300" r="230" fill="none" stroke="rgba(169,159,240,0.12)" />
          </svg>
          <svg viewBox="0 0 600 600" className="absolute h-[470px] w-[470px] animate-spin-slower opacity-60">
            <circle cx="300" cy="300" r="290" fill="none" stroke="rgba(217,70,239,0.18)" strokeDasharray="60 30 4 30" />
          </svg>
        </div>

        {/* Leader lines */}
        <svg className="pointer-events-none absolute inset-0 z-[15] hidden h-full w-full md:block" aria-hidden="true">
          {lines.map((l) => (
            <g key={l.key} stroke="rgba(243,241,248,0.45)" fill="#0c0b12">
              <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} strokeWidth="1" />
              <circle cx={l.x1} cy={l.y1} r="3.5" strokeWidth="1" />
              <circle cx={l.x2} cy={l.y2} r="3.5" strokeWidth="1" />
            </g>
          ))}
        </svg>

        {/* Centre: orb dropzone or the scan */}
        <div className="relative z-10 flex w-full justify-center md:absolute md:inset-0 md:items-center">
          {!preview ? (
            <button
              ref={targetRef}
              type="button"
              onClick={() => inputRef.current?.click()}
              className={`group relative flex h-72 w-72 flex-col items-center justify-center rounded-full border text-center transition-all ${
                dragging ? 'scale-105 border-violet' : 'border-line-strong hover:scale-[1.02]'
              }`}
              style={{
                background:
                  'radial-gradient(circle at 32% 28%, rgba(217,70,239,0.40), rgba(144,133,233,0.28) 38%, rgba(23,21,31,0.85) 72%)',
                boxShadow: '0 0 120px rgba(144,133,233,0.35), inset 0 0 60px rgba(255,255,255,0.06)',
              }}
            >
              <span className="mb-3 rounded-full border border-white/20 bg-white/10 p-3 text-ink">
                <Upload className="h-5 w-5" />
              </span>
              <span className="text-[15px] text-ink">Drop a scan</span>
              <span className="mt-1 px-10 text-xs text-ink-2">or click to browse, or paste</span>
            </button>
          ) : (
            <div ref={targetRef} className="w-[min(100%,360px)] md:w-[34%] md:min-w-[260px] md:max-w-[380px]">
              {visual()}
            </div>
          )}
        </div>

        {/* Callouts */}
        <div className="relative z-20 grid w-full gap-3 sm:grid-cols-2 md:static md:block">
          {!preview && (
            <>
              <Callout slot="tl" cardRef={refs.tl} title="Chest X-ray" sub="14 findings · NIH ChestX-ray14" />
              <Callout slot="tr" cardRef={refs.tr} title="Skin lesion" sub="7 classes · HAM10000 dermoscopy" accent="#d946ef" />
              <Callout slot="bl" cardRef={refs.bl} title="Brain MRI" sub="4 classes · tumour MRI" accent="#199e70" />
            </>
          )}
          {preview && !result && (
            <Callout
              slot="tl"
              cardRef={refs.tl}
              eyebrow={loading ? 'Analyzing' : 'Ready'}
              title={fileName || 'Pasted image'}
              sub={loading ? 'Running the encoders and fusion' : 'Add details, then analyze'}
            />
          )}
          {preview && result && (
            <>
              <Callout
                slot="tl"
                cardRef={refs.tl}
                eyebrow="Modality"
                title={MODALITIES[result.modality]?.label}
                sub={`${SOURCE_LABELS[result.modality_source]}${
                  result.modality_confidence != null ? `, ${pct(result.modality_confidence, 1)}` : ''
                }`}
              />
              {preds[0] && (
                <FindingCallout
                  p={preds[0]}
                  slot="tr"
                  cardRef={refs.tr}
                  override={override}
                  eyebrow={heatmap ? 'Top finding · Grad-CAM focus' : 'Top finding'}
                />
              )}
              {preds[1] && <FindingCallout p={preds[1]} slot="bl" cardRef={refs.bl} eyebrow="Second" />}
              {preds[2] && <FindingCallout p={preds[2]} slot="br" cardRef={refs.br} eyebrow="Third" />}
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="no-print mt-4 flex flex-wrap items-center justify-center gap-3">
        {preview && (
          <>
            <button type="button" onClick={() => inputRef.current?.click()} className="pill pill-sm">
              Replace
            </button>
            <button
              type="button"
              onClick={() => {
                onImage(null)
                if (inputRef.current) inputRef.current.value = ''
              }}
              className="pill pill-sm"
            >
              <Close className="h-3 w-3" /> Remove
            </button>
          </>
        )}
        {heatmap && (
          <>
            <div className="toggle-group">
              {['overlay', 'original'].map((k) => (
                <button key={k} type="button" onClick={() => setView(k)} className={`toggle ${view === k ? 'toggle-on' : ''}`}>
                  {k === 'overlay' ? 'Grad-CAM' : 'Original'}
                </button>
              ))}
            </div>
            {view === 'overlay' && (
              <label className="flex items-center gap-2 text-[11px] uppercase tracking-[0.12em] text-ink-muted">
                Heat
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="w-24 accent-[#9085e9]"
                />
              </label>
            )}
            <button type="button" onClick={() => setFull(true)} className="pill pill-sm" aria-label="Full screen">
              <Expand className="h-3 w-3" /> Expand
            </button>
          </>
        )}
      </div>
      {heatmap && (
        <p className="mx-auto mt-3 max-w-xl text-center text-[11px] leading-relaxed text-ink-muted">
          Grad-CAM for {prettyLabel(result.explanation.heatmap_class)}: warm regions raised its score. It shows what the
          network used, not where the disease is.
        </p>
      )}
      {error && <p className="mt-2 text-center text-xs text-status-serious">{error}</p>}
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => accept(e.target.files?.[0])} />

      {full && (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-black/95 p-6" role="dialog" aria-modal="true">
          <button type="button" onClick={() => setFull(false)} className="pill pill-sm absolute right-6 top-6">
            <Close className="h-3 w-3" /> Close
          </button>
          <div className="h-[80vh]" style={{ aspectRatio: `${ratio}` }}>
            {visual(true)}
          </div>
        </div>
      )}
    </div>
  )
}
