const MODALITY_COLORS = {
  chest: 'bg-blue-100 text-blue-800',
  skin: 'bg-orange-100 text-orange-800',
  brain: 'bg-purple-100 text-purple-800',
}

function getModality(label, ranges) {
  if (!ranges) return null
  for (const [mod, [start, end]] of Object.entries(ranges)) {
    // not used directly — fallback via label index
  }
  return null
}

function getBarColor(score) {
  if (score >= 0.8) return 'bg-red-500'
  if (score >= 0.5) return 'bg-orange-500'
  if (score >= 0.3) return 'bg-yellow-500'
  return 'bg-green-500'
}

export default function PredictionResults({ result }) {
  if (!result) return null

  const { predictions, mode, detected_modality, inference_time_ms } = result

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Predictions</h2>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          {detected_modality && (
            <span
              className={`px-2 py-1 rounded-full font-medium ${
                MODALITY_COLORS[detected_modality] || 'bg-gray-100 text-gray-700'
              }`}
            >
              {detected_modality}
            </span>
          )}
          <span className="bg-gray-100 px-2 py-1 rounded-full">{mode}</span>
          {inference_time_ms && (
            <span>{Math.round(inference_time_ms)}ms</span>
          )}
        </div>
      </div>

      {predictions && predictions.length > 0 ? (
        <div className="space-y-3">
          {predictions.map((pred, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="w-48 text-sm font-medium text-gray-700 truncate" title={pred.disease || pred.label}>
                {(pred.disease || pred.label || '').replace(/_/g, ' ')}
              </div>
              <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getBarColor(
                    pred.confidence
                  )}`}
                  style={{ width: `${Math.max(pred.confidence * 100, 1)}%` }}
                />
              </div>
              <div className="w-16 text-right text-sm font-mono text-gray-600">
                {(pred.confidence * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500">
          No predictions above the confidence threshold.
        </p>
      )}
    </div>
  )
}
