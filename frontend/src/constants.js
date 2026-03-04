/**
 * Constantes partagées du frontend.
 */

export const GRID_STRATEGIES = new Set([
  'grid_atr',
  'grid_boltrend',
  'grid_multi_tf',
  'grid_range_atr',
  'grid_trend',
  'grid_funding',
  'grid_momentum',
  'envelope_dca',
  'envelope_dca_short',
])

export const GRADE_ORDER = { A: 5, B: 4, C: 3, D: 2, F: 1 }

export const SCANNER_COLUMNS = {
  SYMBOL: 'symbol',
  CHANGE: 'change',
  DIR: 'dir',
  GRADE: 'grade',
  DIST_SMA: 'dist_sma',
}
