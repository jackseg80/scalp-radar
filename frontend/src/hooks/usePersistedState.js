/**
 * Hook pour persister un état dans localStorage
 * Usage: const [value, setValue] = usePersistedState('key', defaultValue)
 */

import { useState, useEffect, useRef } from 'react'

export function usePersistedState(key, defaultValue) {
  const storageKey = `scalp-radar-${key}`

  // Charger la valeur initiale depuis localStorage
  const [state, setState] = useState(() => {
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved !== null) {
        const parsed = JSON.parse(saved)
        // Fusionner avec defaultValue pour gérer les nouveaux champs
        if (typeof defaultValue === 'object' && defaultValue !== null && !Array.isArray(defaultValue)) {
          return { ...defaultValue, ...parsed }
        }
        return parsed
      }
    } catch (err) {
      console.warn(`Erreur lecture localStorage ${storageKey}:`, err)
    }
    return defaultValue
  })

  // Sauvegarder dans localStorage à chaque changement
  const isFirstRender = useRef(true)
  useEffect(() => {
    // Skip le premier render (valeur déjà chargée)
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }

    try {
      localStorage.setItem(storageKey, JSON.stringify(state))
    } catch (err) {
      console.warn(`Erreur écriture localStorage ${storageKey}:`, err)
    }
  }, [state, storageKey])

  return [state, setState]
}

/**
 * Hook pour persister plusieurs états liés dans un seul objet localStorage
 * Usage: const [state, updateState] = usePersistedObject('page-key', { field1: val1, field2: val2 })
 */
export function usePersistedObject(key, defaultState) {
  const [state, setState] = usePersistedState(key, defaultState)

  // Helper pour update partiel (comme setState avec objet)
  const updateState = (updates) => {
    if (typeof updates === 'function') {
      setState(prev => ({ ...prev, ...updates(prev) }))
    } else {
      setState(prev => ({ ...prev, ...updates }))
    }
  }

  return [state, updateState, setState]
}
