/**
 * Parse markdown headings (H1-H3) into a tree structure.
 * Strips fenced code blocks first to avoid false matches.
 */
export function parseHeadings(markdown) {
  // Strip fenced code blocks to avoid matching # inside code
  const stripped = markdown.replace(/^```[\s\S]*?^```/gm, '')

  const headingRegex = /^(#{1,3})\s+(.+)$/gm
  const headings = []
  let match

  while ((match = headingRegex.exec(stripped)) !== null) {
    headings.push({
      level: match[1].length,
      text: match[2].trim(),
    })
  }

  if (headings.length === 0) return null

  const root = {
    id: 'h-0',
    level: headings[0].level,
    text: headings[0].text,
    children: [],
  }

  const stack = [root]

  for (let i = 1; i < headings.length; i++) {
    const node = {
      id: `h-${i}`,
      level: headings[i].level,
      text: headings[i].text,
      children: [],
    }

    // Walk up stack to find parent (node with lower level)
    while (stack.length > 1 && stack[stack.length - 1].level >= node.level) {
      stack.pop()
    }

    stack[stack.length - 1].children.push(node)
    stack.push(node)
  }

  return root
}
