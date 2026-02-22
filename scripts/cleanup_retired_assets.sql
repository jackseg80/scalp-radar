-- ============================================================
-- Nettoyage DB : suppression des 14 assets retirés
-- 8 originaux : ENJ, SUSHI, IMX, SAND, APE, XTZ, JUP, AR
-- 6 extras    : INJ, ONDO, SEI, THETA, TIA, WIF
-- ============================================================
-- USAGE (serveur Linux) :
--   cd ~/scalp-radar
--   docker compose stop   # arrêter le bot
--   cp data/scalp_radar.db data/scalp_radar.db.bak.pre_cleanup
--   sqlite3 data/scalp_radar.db < scripts/cleanup_retired_assets.sql
--   docker compose up -d  # relancer le bot
-- ============================================================

-- wfo_combo_results (FK vers optimization_results) — AVANT optimization_results
DELETE FROM wfo_combo_results
WHERE optimization_result_id IN (
    SELECT id FROM optimization_results
    WHERE asset IN (
        'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
        'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
    )
);

DELETE FROM candles WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM funding_rates WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM open_interest WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM position_events WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM simulation_trades WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM signals WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM trades WHERE symbol IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM optimization_results WHERE asset IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

DELETE FROM optimization_jobs WHERE asset IN (
    'ENJ/USDT','SUSHI/USDT','IMX/USDT','SAND/USDT','APE/USDT','XTZ/USDT','JUP/USDT','AR/USDT',
    'INJ/USDT','ONDO/USDT','SEI/USDT','THETA/USDT','TIA/USDT','WIF/USDT'
);

VACUUM;
