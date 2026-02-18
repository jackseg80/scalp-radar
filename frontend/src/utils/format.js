/**
 * Formate un prix crypto avec le bon nombre de décimales selon sa magnitude.
 * BTC (~97 000$) → 2 décimales, ETH (~2 500$) → 2, SOL (~150$) → 2
 * DOGE (~0.30$) → 4, SHIB (0.00002$) → 6
 */
export function formatPrice(price) {
  if (price == null || isNaN(price)) return "—";
  const p = Number(price);
  if (p >= 100) return p.toFixed(2);
  if (p >= 1)   return p.toFixed(4);
  if (p >= 0.01) return p.toFixed(5);
  return p.toFixed(6);
}

/**
 * Formate un P&L avec signe + couleur CSS class.
 * Retourne { text, className }
 */
export function formatPnl(value) {
  if (value == null || isNaN(value)) return { text: "—", className: "" };
  const v = Number(value);
  const sign = v >= 0 ? "+" : "";
  return {
    text: `${sign}${v.toFixed(2)}$`,
    className: v >= 0 ? "text-green-400" : "text-red-400",
  };
}
