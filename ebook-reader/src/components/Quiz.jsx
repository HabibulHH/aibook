import { useState } from 'react'

export default function Quiz({ questions }) {
  const [answers, setAnswers] = useState({})
  const [submitted, setSubmitted] = useState(false)

  const allAnswered = questions.every((_, i) => answers[i] !== undefined)
  const score = questions.filter((q, i) => answers[i] === q.correct).length
  const percent = Math.round((score / questions.length) * 100)

  const reset = () => {
    setAnswers({})
    setSubmitted(false)
  }

  const select = (qi, oi) => {
    if (submitted) return
    setAnswers({ ...answers, [qi]: oi })
  }

  return (
    <div className="not-prose my-10 p-6 md:p-8 rounded-2xl border border-neutral-300 dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-900">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-neutral-900 dark:text-white flex items-center gap-2">
          <span>🎯</span>
          <span>Interactive Quiz</span>
        </h3>
        {submitted && (
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
              Score
            </span>
            <span
              className={`text-sm font-bold px-3 py-1 rounded-full ${
                percent >= 70
                  ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300'
                  : percent >= 40
                  ? 'bg-yellow-100 dark:bg-yellow-900/40 text-yellow-700 dark:text-yellow-300'
                  : 'bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300'
              }`}
            >
              {score} / {questions.length} ({percent}%)
            </span>
          </div>
        )}
      </div>

      <div className="space-y-8">
        {questions.map((q, qi) => (
          <div key={qi}>
            <p className="font-medium mb-3 text-neutral-900 dark:text-white">
              <span className="text-neutral-500 dark:text-neutral-400 mr-2">
                {qi + 1}.
              </span>
              {q.q}
            </p>

            <div className="space-y-2">
              {q.options.map((opt, oi) => {
                const isSelected = answers[qi] === oi
                const isCorrect = oi === q.correct
                const showResult = submitted

                let cls =
                  'w-full text-left px-4 py-3 rounded-lg border transition-all text-sm md:text-base '

                if (showResult) {
                  if (isCorrect) {
                    cls +=
                      'bg-green-50 dark:bg-green-900/30 border-green-500 dark:border-green-600 text-green-900 dark:text-green-100'
                  } else if (isSelected) {
                    cls +=
                      'bg-red-50 dark:bg-red-900/30 border-red-500 dark:border-red-600 text-red-900 dark:text-red-100'
                  } else {
                    cls +=
                      'bg-white dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700 text-neutral-500 dark:text-neutral-500'
                  }
                } else if (isSelected) {
                  cls +=
                    'bg-blue-50 dark:bg-blue-900/30 border-blue-500 dark:border-blue-600 text-neutral-900 dark:text-white'
                } else {
                  cls +=
                    'bg-white dark:bg-neutral-800 border-neutral-200 dark:border-neutral-700 text-neutral-700 dark:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-700 cursor-pointer'
                }

                return (
                  <button
                    key={oi}
                    onClick={() => select(qi, oi)}
                    disabled={submitted}
                    className={cls}
                  >
                    <span className="flex items-center justify-between gap-3">
                      <span>{opt}</span>
                      {showResult && isCorrect && (
                        <span className="text-green-600 dark:text-green-400 font-bold">
                          ✓
                        </span>
                      )}
                      {showResult && isSelected && !isCorrect && (
                        <span className="text-red-600 dark:text-red-400 font-bold">
                          ✗
                        </span>
                      )}
                    </span>
                  </button>
                )
              })}
            </div>

            {submitted && q.why && (
              <div className="mt-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-400 dark:border-blue-600 text-sm text-neutral-700 dark:text-neutral-300">
                <span className="font-semibold text-blue-700 dark:text-blue-300">
                  Why:
                </span>{' '}
                {q.why}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-3 mt-8 pt-6 border-t border-neutral-200 dark:border-neutral-700">
        {!submitted ? (
          <button
            onClick={() => setSubmitted(true)}
            disabled={!allAnswered}
            className="px-5 py-2 rounded-lg bg-neutral-900 dark:bg-white text-white dark:text-neutral-900 font-medium hover:opacity-90 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Submit ({Object.keys(answers).length}/{questions.length})
          </button>
        ) : (
          <button
            onClick={reset}
            className="px-5 py-2 rounded-lg border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-200 font-medium hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
          >
            Try Again
          </button>
        )}
      </div>
    </div>
  )
}
