// Attention-based word importance for the patient-details text.
export default function TextExplanation({ explanation }) {
  const tokens = explanation?.important_tokens || []
  const highlighted = explanation?.highlighted_text
  if (!highlighted && tokens.length === 0) return null
  const max = Math.max(...tokens.map((t) => t.importance), 1e-6)

  return (
    <section className="card card-pad">
      <h2 className="card-title">Which words the text encoder attended to</h2>
      {highlighted && (
        // Rendered as React text nodes, never as HTML: the text is user input.
        <p className="mt-3 rounded-xl border border-line bg-surface-2 p-3 text-sm leading-relaxed text-ink-2">
          {highlighted.split(/\*\*(.*?)\*\*/g).map((part, i) =>
            i % 2 === 1 ? (
              <mark key={i} className="rounded bg-accent-soft px-0.5 text-ink">
                {part}
              </mark>
            ) : (
              part
            ),
          )}
        </p>
      )}
      {tokens.length > 0 && (
        <ul className="mt-4 space-y-2">
          {tokens.map((t, i) => (
            <li key={i} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3 text-xs">
              <span className="truncate font-mono text-ink">{t.token}</span>
              <span className="h-1.5 rounded-full bg-surface-3">
                <span
                  className="block h-1.5 rounded-full bg-accent"
                  style={{ width: `${(t.importance / max) * 100}%` }}
                />
              </span>
              <span className="num text-right text-ink-2">{(t.importance * 100).toFixed(0)}%</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-[11px] text-ink-muted">
        Last-layer [CLS] attention, merged over word pieces and renormalised. Attention is a hint about what was read,
        not proof of why a score is high.
      </p>
    </section>
  )
}
