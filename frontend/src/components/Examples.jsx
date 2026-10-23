import { MODALITIES } from '../lib/domain'
import { exampleLabel, galleryUrl } from '../lib/gallery'

// One-click example cases from the held-out test splits (scripts/build_gallery.py).
export default function Examples({ cases, activeId, onPick, busy }) {
  if (!cases.length) return null
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="eyebrow">Example cases</span>
        <span className="text-[11px] text-ink-muted">Held-out test split · one per class, drawn at random</span>
      </div>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
        {cases.map((c) => {
          const on = c.id === activeId
          return (
            <button
              key={c.id}
              type="button"
              disabled={busy}
              onClick={() => onPick(c)}
              aria-pressed={on}
              title={`${MODALITIES[c.modality].label}, labelled ${exampleLabel(c)}${c.text ? `. ${c.text}` : ''}`}
              className={`group relative aspect-square overflow-hidden rounded-xl border bg-black transition-colors disabled:cursor-wait ${
                on ? 'border-violet ring-2 ring-violet/30' : 'border-line hover:border-ink-2'
              }`}
            >
              <img
                src={galleryUrl(c.thumb)}
                alt=""
                loading="lazy"
                className="h-full w-full object-cover opacity-80 transition-opacity group-hover:opacity-100"
              />
              <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent px-1.5 pb-1.5 pt-5 text-left">
                <span className="block text-[9px] uppercase tracking-[0.12em] text-ink-muted">
                  {MODALITIES[c.modality].short}
                </span>
                <span className="line-clamp-2 block text-[11px] leading-tight text-ink [overflow-wrap:anywhere]">{exampleLabel(c)}</span>
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
