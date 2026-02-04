 cols_cards = [
    "id",                # id del partido
    "competition_id",
    "date_unix",
    "home_name",
    "away_name",
    "refereeID",
    "stadium_name",
    "stadium_location",

    # Conteo de tarjetas globales
    "team_a_yellow_cards",
    "team_b_yellow_cards",
    "team_a_red_cards",
    "team_b_red_cards",
    "team_a_cards_num",
    "team_b_cards_num",

    # Tarjetas por mitades
    "team_a_fh_cards",
    "team_b_fh_cards",
    "team_a_2h_cards",
    "team_b_2h_cards",
    "total_fh_cards",
    "total_2h_cards",

    # Tarjetas en primeros 10 min (señal temprana)
    "team_a_cards_0_10_min",
    "team_b_cards_0_10_min",

    # Señales de juego que correlacionan con tarjetas
    "team_a_fouls",
    "team_b_fouls",
    "team_a_possession",
    "team_b_possession",
    "attacks_recorded",
    "team_a_dangerous_attacks",
    "team_b_dangerous_attacks",

    # Potenciales calculados
    "cards_potential",
    "avg_potential"
]
        df_normalized = pd.DataFrame(matches_raw)
        df_cards = df_normalized[cols_cards]

        print(df_cards)