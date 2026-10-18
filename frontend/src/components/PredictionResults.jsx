const MODALITY_COLORS = {
  chest: 'bg-blue-100 text-blue-800',
  skin: 'bg-orange-100 text-orange-800',
  brain: 'bg-purple-100 text-purple-800',
}

const SOURCE_LABELS = {
  user: 'selected manually',
  classifier: 'auto-detected',
  heuristic: 'auto-detected (heuristic)',
  text: 'inferred from symptoms',
}

const MODE_LABELS = {
  image_only: 'Image only',
  text_only: 'Text only',
  multimodal: 'Multimodal',
}

function getBarColor(score) {
  if (score >= 0.8) return 'bg-red-500'
  if (score >= 0.5) return 'bg-orange-500'
  if (score >= 0.3) return 'bg-yellow-500'
  return 'bg-green-500'
}

export default function PredictionResults({ result }) {
  if (!result) return null

  const {
    predictions,
    mode,
    modality,
    modality_source,
    modality_confidence,
    inference_time_ms,
  } = result

  const sourceLabel = SOURCE_LABELS[modality_source] || ''
  const modalityTitle =
    modality_confidence != null
      ? `${sourceLabel}, ${(modality_confidence * 100).toFixed(0)}% confidence`
      : sourceLabel

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4 gap-2 flex-wrap">
        <h2 className="text-lg font-semibold text-gray-800">Predictions</h2>
        <div className="flex items-center gap-2 text-xs text-gray-500 flex-wrap">
          {modality && (
            <span
              title={modalityTitle}
              className={`px-2 py-1 rounded-full font-medium ${
                MODALITY_COLORS[modality] || 'bg-gray-100 text-gray-700'
              }`}
            >
              {modality}
              {sourceLabel && <span className="font-normal opacity-75"> · {sourceLabel}</span>}
            </span>
          )}
          <span className="bg-gray-100 px-2 py-1 rounded-full">{MODE_LABELS[mode] || mode}</span>
          {inference_time_ms != null && <span>{Math.round(inference_time_ms)} ms</span>}
        </div>
      </div>

      {predictions && predictions.length > 0 ? (
        <div className="space-y-3">
          {predictions.map((pred) => (
            <div key={pred.label_index} className="flex items-center gap-3">
              <div
                className="w-48 text-sm font-medium text-gray-700 truncate"
                title={pred.disease}
              >
                {pred.disease.replace(/_/g, ' ')}
              </div>
              <div className="relative flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getBarColor(
                    pred.confidence
                  )}`}
                  style={{ width: `${Math.max(pred.confidence * 100, 1)}%` }}
                />
                {/* decision threshold marker */}
                <div
                  className="absolute top-0 h-full w-0.5 bg-gray-700/60"
                  style={{ left: `${pred.threshold * 100}%` }}
                  title={`Decision threshold ${(pred.threshold * 100).toFixed(0)}%`}
                />
              </div>
              <div className="w-16 text-right text-sm font-mono text-gray-600">
                {(pred.confidence * 100).toFixed(1)}%
              </div>
              <div className="w-20 text-right">
                {pred.positive && (
                  <span className="text-[11px] font-semibold text-red-700 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded">
                    positive
                  </span>
                )}
              </div>
            </div>
          ))}
          <p className="text-[11px] text-gray-400 pt-1">
            {result.calibrated
              ? 'Scores are temperature-calibrated on validation data.'
              : 'Scores are uncalibrated model outputs.'}{' '}
            The tick on each bar marks that class&apos;s decision threshold.
          </p>
        </div>
      ) : (
        <p className="text-sm text-gray-500">No predictions returned.</p>
      )}
    </div>
  )
}
