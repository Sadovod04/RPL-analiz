"""RU/EN strings, position names, feature labels for the dashboard."""

from __future__ import annotations

LANGS = ("ru", "en")

# --- coarse position bucket -> label ---------------------------------
POSITION_RU = {
    "GK": "Вратарь",
    "CB": "Центр. защитник",
    "FB": "Крайний защитник",
    "CM": "Полузащитник",
    "W": "Вингер",
    "ST": "Нападающий",
    "UNKNOWN": "—",
}
POSITION_EN = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back",
    "CM": "Midfielder",
    "W": "Winger",
    "ST": "Forward",
    "UNKNOWN": "—",
}

# --- fine Transfermarkt position -> RU (abbr + full) ------------------
POSITION_DETAIL_RU = {
    "Goalkeeper": ("ВРТ", "Вратарь"),
    "Sweeper": ("ЛИБ", "Либеро"),
    "Centre-Back": ("ЦЗ", "Центральный защитник"),
    "Left-Back": ("ЛЗ", "Левый защитник"),
    "Right-Back": ("ПЗ", "Правый защитник"),
    "Defensive Midfield": ("ОПЗ", "Опорный полузащитник"),
    "Central Midfield": ("ЦП", "Центральный полузащитник"),
    "Attacking Midfield": ("ЦАП", "Центральный атакующий полузащитник"),
    "Left Midfield": ("ЛП", "Левый полузащитник"),
    "Right Midfield": ("ПП", "Правый полузащитник"),
    "Left Winger": ("ЛВ", "Левый вингер"),
    "Right Winger": ("ПВ", "Правый вингер"),
    "Second Striker": ("ОТТ", "Оттянутый форвард"),
    "Centre-Forward": ("НАП", "Центральный нападающий"),
}


def position_label(coarse: str | None, detail: str | None, lang: str) -> str:
    if detail and detail in POSITION_DETAIL_RU:
        abbr, full = POSITION_DETAIL_RU[detail]
        return f"{abbr} · {full}" if lang == "ru" else detail
    table = POSITION_RU if lang == "ru" else POSITION_EN
    return table.get(str(coarse), str(coarse or "—"))


# --- feature names -> human label -----------------------------------
FEATURE_LABELS = {
    "ru": {
        "youth_minutes_total": "Сыграно минут в молодёжке (до отсечки)",
        "youth_goals_total": "Забито голов в молодёжке",
        "youth_ga_per90": "Гол+пас за 90 минут (в молодёжке)",
        "youth_minutes_trend": "Динамика игрового времени: + роль росла, − падала",
        "youth_seasons": "Сколько сезонов в молодёжке",
        "rpl_minutes_ever": "Минут в РПЛ за карьеру",
        "best_level_pre_cutoff": "Макс. лига до отсечки (0 нет · 1 ЮФЛ · 2 ФНЛ-2 · 3 ФНЛ · 4 РПЛ)",
        "played_youth_league": "Играл в ЮФЛ",
        "minutes_U13": "Минуты в возрасте до 13",
        "minutes_U15": "Минуты в 13–14",
        "minutes_U17": "Минуты в 15–16",
        "minutes_U19": "Минуты в 17–18",
        "minutes_U21": "Минуты в 19–20",
        "ga_per90_U13": "Гол+пас/90 до 13",
        "ga_per90_U15": "Гол+пас/90 в 13–14",
        "ga_per90_U17": "Гол+пас/90 в 15–16",
        "ga_per90_U19": "Гол+пас/90 в 17–18",
        "ga_per90_U21": "Гол+пас/90 в 19–20",
        "academy_conversion_rate": "Доля выпускников академии, дошедших до РПЛ (по прошлым наборам)",
        "market_value_at_cutoff_eur": "Оценка стоимости на момент отсечки, €",
        "height_cm": "Рост, см",
        "is_foreigner": "Легионер",
        "position": "Позиция (группа)",
        "position_detail": "Позиция (детально)",
    },
    "en": {},  # falls back to the raw column name
}


def feat_label(col: str, lang: str) -> str:
    return FEATURE_LABELS.get(lang, {}).get(col, col)


# --- UI strings ----------------------------------------------------
STRINGS = {
    "ru": {
        "title": "RPL-analiz — прорыв в РПЛ",
        "subtitle": (
            "Оценка вероятности, что игрок дойдёт до проф. футбола (РПЛ / ФНЛ / ФНЛ-2), "
            "по данным его молодёжной карьеры до отсечки. Это ранжирование потока, а не "
            "вердикт — тема шумная (травмы, форма, решения тренера в цифрах не видны)."
        ),
        "lang": "Язык",
        "tab_pro": "Уже в РПЛ / проф.",
        "tab_prospects": "Перспективные",
        "filters": "Фильтры",
        "position": "Позиция",
        "academy": "Академия",
        "top_n": "Строк в таблице",
        "top_n_help": "Сколько игроков показать в таблице (уже отсортированы по оценке).",
        "all_ph": "все",
        "birth_range": "Год рождения",
        "pos_help": "Оставить только выбранные позиции. Пусто = все.",
        "lvl_help": "До какого уровня игрок реально дошёл в карьере. Пусто = все.",
        "yr_help": "Диапазон годов рождения.",
        "ac_help": "Академия игрока (из данных Transfermarkt, у части игроков пусто).",
        "level_reached": "Достигнутый уровень",
        "col_level_reached": "Уровень",
        "tab_compare": "Сравнение",
        "compare_hdr": "Сравнить игроков",
        "compare_pick": "Выбери 2–5 игроков",
        "compare_need": "Выбери хотя бы двух игроков для сравнения.",
        "compare_stats": "Показатели",
        "compare_chart": "Ключевые метрики молодёжки",
        "compare_minutes": "Минуты (молодёжка + карьера)",
        "compare_rates": "Результативность (гол+пас за 90)",
        "compare_norm": "Сравнение по ключевым метрикам (% от лучшего в группе)",
        "tab_youth": "Юные (ФФ СПб)",
        "youth_hdr": "Юные футболисты СПб (10–13 лет)",
        "youth_note": "Данные ФФ СПб (детские турниры). У этих детей ещё нет карьеры для модели — это отдельный пул талантов: сортируй по голам за игру и объёму игр.",
        "youth_none": "Файл data/processed/ffspb_players.parquet не найден. Запусти scripts/ingest_ffspb.py.",
        "y_name": "Игрок",
        "y_patr": "Отчество",
        "y_teams": "Команды",
        "y_trn": "Турниров",
        "y_games": "Игр",
        "y_goals": "Голов",
        "y_gpg": "Голов/игра",
        "y_min_games": "Мин. игр",
        "y_search": "Поиск по фамилии",
        "target_help": (
            "**Только РПЛ** — вероятность, что игрок дорастёт именно до Премьер-лиги "
            "(≥200 минут в РПЛ). Редкий исход (~20%), жёсткий отбор.\n\n"
            "**Проф. уровень** — вероятность дойти до профессионального футбола вообще: "
            "РПЛ, ФНЛ или ФНЛ-2. Частый исход (~84%), модель точнее."
        ),
        "report_hdr": "Вывод по игроку",
        "report_level": "Предполагаемый уровень",
        "report_strengths": "Сильные стороны",
        "report_weaknesses": "Слабые стороны",
        "lvl_rpl_high": "высокий шанс РПЛ",
        "lvl_rpl_mid": "есть шанс на РПЛ",
        "lvl_fnl": "скорее уровень ФНЛ / Первой лиги",
        "lvl_fnl2": "уровень ФНЛ-2 / низших лиг",
        "lvl_low": "низкий шанс профессионального уровня",
        "s_minutes": "много игрового времени в молодёжке",
        "s_output": "высокая результативность (гол+пас)",
        "s_trend": "роль в команде растёт год к году",
        "s_level": "уже играл на высоком уровне для своего возраста",
        "s_youthleague": "прошёл через ЮФЛ",
        "s_value": "высокая оценка рыночной стоимости на момент отсечки",
        "s_academy": "сильная академия по исторической конверсии",
        "w_minutes": "мало игрового времени в молодёжке",
        "w_output": "низкая результативность",
        "w_trend": "роль в команде снижается",
        "w_level": "не поднимался выше юношеского уровня",
        "w_nodata": "мало данных для уверенного вывода",
        "prospects_hdr": "Топ перспективных",
        "pro_hdr": "Игроки с решённой судьбой",
        "why": "Разбор игрока",
        "pick_player": "Игрок",
        "no_match": "Нет игроков под фильтры",
        "score": "Оценка прорыва, %",
        "col_name": "Игрок",
        "col_birth": "Г.р.",
        "col_pos": "Позиция",
        "col_academy": "Академия",
        "col_minutes": "Мол. минуты",
        "col_ga90": "Г+П/90",
        "col_level": "Макс. уровень",
        "raw_stats": "Его статистика",
        "shap_hdr": "Что двигает оценку",
        "shap_help": (
            "Вклад каждого признака в оценку относительно среднего игрока выборки. "
            "Плюс — тянет оценку вверх, минус — вниз, ноль — не влияет."
        ),
        "similar_hdr": "Профиль похож на заигравших",
        "similar_help": (
            "Игроки, которые УЖЕ дошли до проф. уровня и чья молодёжная статистика "
            "по годам ближе всего к этому игроку (та же позиция). Чем меньше «расстояние», "
            "тем похожее."
        ),
        "similar_dist": "расстояние",
        "similar_none": "Похожих заигравших не нашлось (мало данных по позиции).",
        "demo_warning": (
            "Загружен демо-датасет (малый, перекошен). Для реальной картины нужен полный "
            "сбор данных по академиям."
        ),
        "single_class": (
            "В загруженных данных среди игроков с решённой судьбой только один класс "
            "(ожидаемо для демо-выборки). Запусти полный сбор данных."
        ),
        "outcome_yes": "дошёл до проф.",
        "outcome_no": "не дошёл",
    },
    "en": {
        "title": "RPL-analiz — breakthrough",
        "subtitle": (
            "Probability a player will reach professional football (RPL / FNL / FNL-2) "
            "from their youth-career data before the cutoff. A ranking of the stream, "
            "not a verdict."
        ),
        "lang": "Language",
        "tab_pro": "Already pro / RPL",
        "tab_prospects": "Prospects",
        "filters": "Filters",
        "position": "Position",
        "academy": "Academy",
        "top_n": "Rows in table",
        "top_n_help": "How many players to show (already sorted by score).",
        "all_ph": "all",
        "birth_range": "Birth year",
        "pos_help": "Keep only the selected positions. Empty = all.",
        "lvl_help": "The level the player actually reached in their career. Empty = all.",
        "yr_help": "Birth-year range.",
        "ac_help": "Player's academy (from Transfermarkt; missing for some players).",
        "level_reached": "Level reached",
        "col_level_reached": "Level",
        "tab_compare": "Compare",
        "compare_hdr": "Compare players",
        "compare_pick": "Pick 2–5 players",
        "compare_need": "Pick at least two players to compare.",
        "compare_stats": "Stats",
        "compare_chart": "Key youth metrics",
        "compare_minutes": "Minutes (youth + career)",
        "compare_rates": "Output (goals+assists per 90)",
        "compare_norm": "Key metrics vs the best in the group (%)",
        "tab_youth": "Youth (SPb FF)",
        "youth_hdr": "SPb youth footballers (age 10–13)",
        "youth_note": "SPb FF data (kids' tournaments). No career yet for the model — a separate talent pool: sort by goals per game and games played.",
        "youth_none": "data/processed/ffspb_players.parquet not found. Run scripts/ingest_ffspb.py.",
        "y_name": "Player",
        "y_patr": "Patronymic",
        "y_teams": "Teams",
        "y_trn": "Tournaments",
        "y_games": "Games",
        "y_goals": "Goals",
        "y_gpg": "Goals/game",
        "y_min_games": "Min games",
        "y_search": "Search by surname",
        "target_help": (
            "**RPL only** — probability of reaching the Premier League specifically "
            "(>=200 RPL minutes). Rare (~20%), a hard bar.\n\n"
            "**Any pro** — probability of reaching professional football at all: RPL, "
            "FNL or FNL-2. Common (~84%), the model is more accurate here."
        ),
        "report_hdr": "Player takeaway",
        "report_level": "Projected level",
        "report_strengths": "Strengths",
        "report_weaknesses": "Weaknesses",
        "lvl_rpl_high": "strong RPL chance",
        "lvl_rpl_mid": "some RPL chance",
        "lvl_fnl": "more like FNL / First League level",
        "lvl_fnl2": "FNL-2 / lower-league level",
        "lvl_low": "low chance of a professional level",
        "s_minutes": "lots of youth playing time",
        "s_output": "high output (goals+assists)",
        "s_trend": "role growing year on year",
        "s_level": "already played at a high level for their age",
        "s_youthleague": "came through the youth league",
        "s_value": "high market-value estimate at cutoff age",
        "s_academy": "strong academy by historical conversion",
        "w_minutes": "little youth playing time",
        "w_output": "low output",
        "w_trend": "role declining",
        "w_level": "did not rise above youth level",
        "w_nodata": "thin data for a confident call",
        "prospects_hdr": "Top prospects",
        "pro_hdr": "Players with a settled outcome",
        "why": "Player breakdown",
        "pick_player": "Player",
        "no_match": "No players match the filters",
        "score": "Breakthrough score, %",
        "col_name": "Player",
        "col_birth": "Born",
        "col_pos": "Position",
        "col_academy": "Academy",
        "col_minutes": "Youth minutes",
        "col_ga90": "G+A/90",
        "col_level": "Peak level",
        "raw_stats": "Their stats",
        "shap_hdr": "What moves the score",
        "shap_help": (
            "Each feature's contribution to the score relative to the average player. "
            "Positive pushes the score up, negative pulls it down, zero has no effect."
        ),
        "similar_hdr": "Profile resembles these breakthrough players",
        "similar_help": (
            "Players who ALREADY reached pro level and whose year-by-year youth stats are "
            "closest to this player (same position). Smaller distance = more similar."
        ),
        "similar_dist": "distance",
        "similar_none": "No similar breakthrough players found (thin data for this position).",
        "demo_warning": (
            "Demo dataset loaded (small, skewed). A full academy-wide crawl is needed for "
            "real numbers."
        ),
        "single_class": (
            "The loaded data has only one outcome class among settled players (expected for "
            "the demo sample). Run the full crawl."
        ),
        "outcome_yes": "reached pro",
        "outcome_no": "did not",
    },
}


def t(key: str, lang: str) -> str:
    return STRINGS.get(lang, STRINGS["ru"]).get(key, key)
