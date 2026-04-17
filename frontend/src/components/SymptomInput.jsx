const EXAMPLE_SYMPTOMS = [
  'A 65-year-old male presents with productive cough, fever, and dyspnea for the past week.',
  'Patient reports an asymmetric mole with irregular borders that has recently changed in size.',
  'A 45-year-old female complains of progressive headaches worse in the morning and new onset seizures.',
]

export default function SymptomInput({ value, onChange }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">
        Patient Symptoms
      </h2>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Describe the patient's symptoms, medical history, and clinical presentation..."
        rows={5}
        className="w-full border border-gray-300 rounded-lg p-3 text-sm
                   focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                   resize-none placeholder:text-gray-400"
      />
      <div className="mt-3">
        <p className="text-xs text-gray-400 mb-2">Quick examples:</p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLE_SYMPTOMS.map((ex, i) => (
            <button
              key={i}
              onClick={() => onChange(ex)}
              className="text-xs bg-gray-100 hover:bg-blue-100 text-gray-600
                         hover:text-blue-700 px-3 py-1.5 rounded-full transition-colors
                         truncate max-w-[260px]"
              title={ex}
            >
              {ex.slice(0, 55)}...
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
