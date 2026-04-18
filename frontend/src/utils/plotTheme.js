export const PLOT_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font:          { color: '#e5e2e1', family: 'Inter, sans-serif', size: 10 },
  margin:        { l: 40, r: 10, t: 40, b: 30 },
}

export const GRID = { gridcolor: 'rgba(81, 67, 65, 0.2)' }

export const COLORS = {
  accent:  '#ffb3b1',  /* Primary */
  light:   '#dfb9b4',  /* Secondary */
  teal:    '#80cbc4',
  blue:    '#81d4fa',
  yellow:  '#ffe082',
  red:     '#ffb4ab',
  muted:   '#aa8986',  /* Outline */
}

export const CATEGORY_COLORS = {
  Structural:   COLORS.accent,
  Textual:      COLORS.blue,
  Evolutionary: COLORS.teal,
  Unknown:      COLORS.muted,
}

export const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
}
