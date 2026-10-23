// Attention-based word importance for the patient-details text.
export default function TextExplanation({ explanation }) {
  const tokens = explanation?.important_tokens || []
  const highlighted = explanation?.highlighted_text
  const max = Math.max(...tokens.map((t) => t.importance), 1e-6)

  return (
    <div className="grid gap-8 md:grid-cols-2">
      {highlighted && (
        <div>
          <div className="eyebrow mb-3">Text read by BioBERT</div>
          {/* Rendered as React text nodes, never as HTML: the text is user input. */}
          <p className="display text-xl leading-relaxed text-ink-2">
            {highlighted.split(/\*\*(.*?)\*\*/g).map((part, i) =>
              i % 2 === 1 ? (
                <mark key={i} className="rounded bg-violet-soft px-1 text-ink">
                  {part}
                </mark>
              ) : (
                part
              ),
            )}
          </p>
        </div>
      )}
      {tokens.length > 0 && (
        <div>
          <div className="eyebrow mb-3">Word importance</div>
          <ul className="space-y-2.5">
            {tokens.map((t, i) => (
              <li key={i} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3 text-xs">
                <span className="truncate font-mono text-ink">{t.token}</span>
                <span className="h-1.5 rounded-full bg-white/[0.07]">
                  <span className="block h-1.5 rounded-full bg-violet-strong" style={{ width: `${(t.importance / max) * 100}%` }} />
                </span>
                <span className="num text-right text-ink-2">{(t.importance * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="text-xs text-ink-muted md:col-span-2">
        Last-layer [CLS] attention, merged over word pieces and renormalised. Attention hints at what was read; it is not
        proof of why a score is high.
      </p>
    </div>
  )
}
