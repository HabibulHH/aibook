const MAX_CHARS = { 1: 30, 2: 26, 3: 24 }

function truncate(text, level) {
  const max = MAX_CHARS[level] || 24
  return text.length > max ? text.slice(0, max - 1) + '…' : text
}

export default function MindMapNode({ x, y, text, level, nodeW, nodeH }) {
  const label = truncate(text, level)

  // Level-based styling
  const styles = {
    1: {
      rect: 'fill-neutral-900 dark:fill-neutral-100',
      text: 'fill-white dark:fill-neutral-900',
      fontSize: 13,
      fontWeight: 700,
      rx: 10,
    },
    2: {
      rect: 'fill-neutral-100 dark:fill-neutral-800 stroke-neutral-300 dark:stroke-neutral-600',
      text: 'fill-neutral-800 dark:fill-neutral-200',
      fontSize: 11.5,
      fontWeight: 600,
      rx: 8,
    },
    3: {
      rect: 'fill-neutral-50 dark:fill-neutral-700 stroke-neutral-200 dark:stroke-neutral-600',
      text: 'fill-neutral-600 dark:fill-neutral-300',
      fontSize: 10.5,
      fontWeight: 400,
      rx: 6,
    },
  }

  const s = styles[level] || styles[3]

  return (
    <g transform={`translate(${x - nodeW / 2}, ${y - nodeH / 2})`}>
      <rect
        width={nodeW}
        height={nodeH}
        rx={s.rx}
        ry={s.rx}
        className={s.rect}
        strokeWidth={level === 1 ? 0 : 1}
      />
      <text
        x={nodeW / 2}
        y={nodeH / 2}
        textAnchor="middle"
        dominantBaseline="central"
        className={s.text}
        fontSize={s.fontSize}
        fontWeight={s.fontWeight}
        fontFamily="'Inter', system-ui, sans-serif"
      >
        {label}
      </text>
    </g>
  )
}
