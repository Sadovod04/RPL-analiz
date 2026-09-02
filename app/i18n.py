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
        "youth_minutes_total": "Всего минут в молодёжке (до отсечки)",
        "youth_goals_total": "Всего голов в молодёжке",
        "youth_ga_per90": "Гол+пас на 90 минут",
        "youth_minutes_trend": "Тренд минут по годам (растёт ли роль)",
        "youth_seasons": "Сезонов в молодёжке",
        "best_level_pre_cutoff": "Макс. уровень до отсечки (0 нет · 1 ЮФЛ · 2 ФНЛ-2 · 3 ФНЛ · 4 РПЛ)",
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
        "min_minutes": "Мин. молодёжных минут",
        "academy": "Академия",
        "top_n": "Показать топ-N",
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
        "min_minutes": "Min youth minutes",
        "academy": "Academy",
        "top_n": "Show top N",
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
