export default function BookCover({ meta, onStart }) {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-md w-full text-center">
        {/* Book cover visual */}
        <div className="mx-auto mb-10 w-64 aspect-[2/3] border-2 border-neutral-900 dark:border-neutral-100 rounded-sm
                        flex flex-col items-center justify-center p-8 relative
                        shadow-[8px_8px_0px_0px_rgba(0,0,0,0.1)] dark:shadow-[8px_8px_0px_0px_rgba(255,255,255,0.05)]">
          {/* Decorative line */}
          <div className="absolute top-6 left-6 right-6 border-t border-neutral-400 dark:border-neutral-500" />
          <div className="absolute bottom-6 left-6 right-6 border-t border-neutral-400 dark:border-neutral-500" />

          <h1 className="font-serif text-2xl font-bold leading-tight mb-2">
            {meta.title}
          </h1>
          {meta.subtitle && (
            <p className="text-xs leading-snug opacity-60 mb-4">
              {meta.subtitle}
            </p>
          )}
          <div className="w-12 border-t border-neutral-900 dark:border-neutral-100 mb-4" />
          <p className="text-sm tracking-widest uppercase opacity-70">
            {meta.author}
          </p>
        </div>

        {/* Description */}
        <p className="text-neutral-600 dark:text-neutral-400 font-serif italic mb-8 leading-relaxed px-4">
          "{meta.description}"
        </p>

        {/* Start reading button */}
        <button
          onClick={onStart}
          className="px-8 py-3 bg-neutral-900 dark:bg-white text-white dark:text-black
                     rounded-full font-medium tracking-wide text-sm
                     hover:bg-neutral-700 dark:hover:bg-neutral-200 transition-colors"
        >
          Begin Reading
        </button>

        <p className="mt-6 text-xs text-neutral-400 dark:text-neutral-500">
          {meta.year} &middot; 9 Chapters
        </p>
      </div>
    </div>
  )
}
