import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Reader({ chapter, totalChapters, onPrev, onNext }) {
  const contentRef = useRef(null)

  useEffect(() => {
    contentRef.current?.scrollTo(0, 0)
  }, [chapter.id])

  return (
    <div ref={contentRef} className="flex-1 overflow-y-auto">
      <article className="max-w-2xl mx-auto px-6 py-12 md:px-10 md:py-16">
        {/* Chapter indicator */}
        <div className="mb-8">
          <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
            Chapter {chapter.id} of {totalChapters}
          </span>
          {/* Progress bar */}
          <div className="mt-3 h-0.5 bg-neutral-200 dark:bg-neutral-700 rounded-full">
            <div
              className="h-full bg-neutral-900 dark:bg-white rounded-full transition-all duration-500"
              style={{ width: `${(chapter.id / totalChapters) * 100}%` }}
            />
          </div>
        </div>

        {/* Markdown content */}
        <div className="prose font-serif text-neutral-800 dark:text-neutral-200 text-base md:text-lg">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {chapter.content}
          </ReactMarkdown>
        </div>

        {/* Navigation */}
        <div className="mt-16 pt-8 border-t border-neutral-200 dark:border-neutral-700 flex justify-between items-center">
          <button
            onClick={onPrev}
            disabled={chapter.id === 1}
            className="flex items-center gap-2 text-sm text-neutral-500 dark:text-neutral-400
                       hover:text-neutral-900 dark:hover:text-white transition-colors
                       disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
            Previous
          </button>

          <span className="text-xs text-neutral-400 dark:text-neutral-500">
            {chapter.id} / {totalChapters}
          </span>

          <button
            onClick={onNext}
            disabled={chapter.id === totalChapters}
            className="flex items-center gap-2 text-sm text-neutral-500 dark:text-neutral-400
                       hover:text-neutral-900 dark:hover:text-white transition-colors
                       disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Next
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </article>
    </div>
  )
}
