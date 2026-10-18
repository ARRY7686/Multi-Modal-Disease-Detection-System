export default function ExplanationPanel({ explanation }) {
  if (!explanation) return null

  const { important_tokens, highlighted_text } = explanation

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">
        Text Explanation
      </h2>

      {highlighted_text && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-600 mb-2">
            Highlighted Symptoms
          </h3>
          {/* Rendered as React text nodes (never as HTML): the text is user input. */}
          <p className="text-sm text-gray-700 leading-relaxed bg-gray-50 p-3 rounded-lg">
            {highlighted_text.split(/\*\*(.*?)\*\*/g).map((part, i) =>
              i % 2 === 1 ? (
                <span key={i} className="bg-yellow-200 font-semibold px-0.5 rounded">
                  {part}
                </span>
              ) : (
                part
              )
            )}
          </p>
        </div>
      )}

      {important_tokens && important_tokens.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-600 mb-2">
            Top Important Tokens
          </h3>
          <div className="flex flex-wrap gap-2">
            {important_tokens.map((tok, i) => {
              const opacity = Math.max(0.3, tok.importance)
              return (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs
                             font-medium border border-blue-200"
                  style={{
                    backgroundColor: `rgba(59, 130, 246, ${opacity * 0.3})`,
                    color: opacity > 0.5 ? '#1e3a5f' : '#4b5563',
                  }}
                >
                  {tok.token}
                  <span className="text-[10px] opacity-70">
                    {(tok.importance * 100).toFixed(0)}%
                  </span>
                </span>
              )
            })}
          </div>
        </div>
      )}

      {!highlighted_text && (!important_tokens || important_tokens.length === 0) && (
        <p className="text-sm text-gray-400">
          No text explanation available for this prediction mode.
        </p>
      )}
    </div>
  )
}
