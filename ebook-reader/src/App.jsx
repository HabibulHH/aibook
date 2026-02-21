import { useState, useEffect } from 'react'
import { bookMeta, chapters } from './bookData'
import Sidebar from './components/Sidebar'
import BookCover from './components/BookCover'
import Reader from './components/Reader'
import ThemeToggle from './components/ThemeToggle'

function App() {
  const [dark, setDark] = useState(() => {
    const saved = localStorage.getItem('ebook-theme')
    if (saved) return saved === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [currentChapter, setCurrentChapter] = useState(null) // null = cover
  const [sidebarOpen, setSidebarOpen] = useState(false)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('ebook-theme', dark ? 'dark' : 'light')
  }, [dark])

  const activeChapter = chapters.find(c => c.id === currentChapter)

  function selectChapter(id) {
    setCurrentChapter(id)
    setSidebarOpen(false)
  }

  function goToCover() {
    setCurrentChapter(null)
    setSidebarOpen(false)
  }

  return (
    <div className="flex h-screen bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 transition-colors">
      {/* Sidebar */}
      <Sidebar
        chapters={chapters}
        currentChapter={currentChapter}
        onSelect={selectChapter}
        onCoverClick={goToCover}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-neutral-200 dark:border-neutral-700 shrink-0">
          <div className="flex items-center gap-3">
            {/* Hamburger menu for mobile */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors"
              aria-label="Open menu"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            <h1 className="font-serif text-sm font-medium truncate">
              {bookMeta.title}
              {activeChapter && (
                <span className="text-neutral-400 dark:text-neutral-500 ml-2">
                  &mdash; {activeChapter.title}
                </span>
              )}
            </h1>
          </div>

          <ThemeToggle dark={dark} onToggle={() => setDark(d => !d)} />
        </header>

        {/* Content area */}
        {currentChapter === null ? (
          <BookCover meta={bookMeta} onStart={() => setCurrentChapter(1)} />
        ) : (
          <Reader
            chapter={activeChapter}
            totalChapters={chapters.length}
            onPrev={() => setCurrentChapter(id => Math.max(1, id - 1))}
            onNext={() => setCurrentChapter(id => Math.min(chapters.length, id + 1))}
          />
        )}
      </div>
    </div>
  )
}

export default App
