/**
 * Tooltip — Composant réutilisable CSS-only avec positionnement auto.
 * Props :
 *   children — élément déclencheur (hover)
 *   content  — texte ou JSX du tooltip
 *   position — "top" (défaut) ou "bottom"
 *   inline   — true pour display inline-flex (défaut), false pour block
 */
import { useRef, useState, useCallback } from 'react'

export default function Tooltip({ children, content, position = 'top', inline = true }) {
  const wrapRef = useRef(null)
  const [pos, setPos] = useState(position)

  const handleEnter = useCallback(() => {
    if (!wrapRef.current) return
    const rect = wrapRef.current.getBoundingClientRect()
    // Si trop près du haut, basculer en bottom
    if (position === 'top' && rect.top < 60) {
      setPos('bottom')
    } else if (position === 'bottom' && window.innerHeight - rect.bottom < 60) {
      setPos('top')
    } else {
      setPos(position)
    }
  }, [position])

  if (!content) return children

  return (
    <span
      ref={wrapRef}
      className={`tooltip-wrap ${inline ? '' : 'tooltip-wrap--block'}`}
      onMouseEnter={handleEnter}
    >
      {children}
      <span className={`tooltip-box tooltip-box--${pos}`}>
        {content}
      </span>
    </span>
  )
}
