export default function Sidebar({ chapters, currentChapter, onSelect, onCoverClick, open, onClose }) {
  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-black/40 z-30 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`
          fixed top-0 left-0 z-40 h-full w-72
          bg-white dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-700
          transform transition-transform duration-300 ease-in-out
          ${open ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0 lg:static lg:z-auto
          flex flex-col
        `}
      >
        {/* Header */}
        <div className="p-5 border-b border-neutral-200 dark:border-neutral-700">
          <button
            onClick={onCoverClick}
            className="text-left w-full group"
          >
            <h2 className="font-serif text-lg font-bold group-hover:underline">
              Contents
            </h2>
          </button>
        </div>

        {/* Chapter list */}
        <nav className="flex-1 overflow-y-auto p-3">
          {/* Cover link */}
          <button
            onClick={onCoverClick}
            className={`
              w-full text-left px-4 py-3 rounded-lg mb-1 text-sm transition-colors
              ${currentChapter === null
                ? "bg-neutral-900 text-white dark:bg-white dark:text-black font-semibold"
                : "hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-600 dark:text-neutral-400"
              }
            `}
          >
            Book Cover
          </button>

          {chapters.map((ch, i) => (
            <button
              key={ch.id}
              onClick={() => onSelect(ch.id)}
              className={`
                w-full text-left px-4 py-3 rounded-lg mb-1 text-sm transition-colors
                ${currentChapter === ch.id
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-black font-semibold"
                  : "hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-600 dark:text-neutral-400"
                }
              `}
            >
              <span className="text-xs opacity-50 mr-2">{String(i + 1).padStart(2, "0")}</span>
              {ch.title}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-neutral-200 dark:border-neutral-700 text-xs text-neutral-400 dark:text-neutral-500">
          eBook Reader
        </div>
      </aside>
    </>
  )
}
