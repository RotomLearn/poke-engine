use std::cmp::max;

use crate::choices::{Choice, Choices, MoveCategory};
use crate::generate_instructions::{
    calculate_damage_rolls, cannot_use_move, move_has_no_effect, moves_first,
};
use crate::instruction::StateInstructions;
use crate::items::Items;
use crate::matchup_mcts::{create_simulation_state, BattleConditions};
use crate::state::{PokemonIndex, SideMovesFirst, SideReference, State};
use crate::state::{PokemonMoveIndex, PokemonType};

// Matchup Classification Thresholds
const STRONG_FAVORABLE_THRESHOLD: f32 = 0.9; // Win percentage for "strongly favorable" matchup (counter)
const FAVORABLE_THRESHOLD: f32 = 0.7; // Win percentage for "favorable" matchup (check)
const UNFAVORABLE_THRESHOLD: f32 = 0.3; // Win percentage for "unfavorable" matchup (checked)
const STRONG_UNFAVORABLE_THRESHOLD: f32 = 0.1; // Win percentage for "strongly unfavorable" matchup (hard countered)

pub fn compute_matchup_fast(
    state: &State,
    s1_idx: PokemonIndex,
    s2_idx: PokemonIndex,
    conditions: &BattleConditions,
) -> i8 {
    // Create a simulation state with the specified conditions
    let mut sim_state = create_simulation_state(state, s1_idx, s2_idx, conditions);

    // Analyze who would win in a direct 1v1 battle
    let win_percentage = analyze_win_percentage(&mut sim_state);

    // Classify the matchup based on win percentage
    classify_matchup_result(win_percentage)
}

/// Analyzes the matchup between the two Pokémon using mathematical win conditions
fn analyze_win_percentage(sim_state: &mut State) -> f32 {
    // Find the best move choices for each Pokémon
    let (s1_best_choice, s1_best_index) = find_best_move(sim_state, &SideReference::SideOne);
    let (s2_best_choice, s2_best_index) = find_best_move(sim_state, &SideReference::SideTwo);

    // Get setup moves if present
    let s1_has_setup = has_setup_move(sim_state, &SideReference::SideOne);
    let s2_has_setup = has_setup_move(sim_state, &SideReference::SideTwo);

    // Analyze speed to determine who goes first
    let mut s1_move_choice = s1_best_choice.clone();
    let mut s2_move_choice = s2_best_choice.clone();
    s1_move_choice.first_move = true;
    s2_move_choice.first_move = true;

    let mut instructions = StateInstructions::default();
    let first_mover = moves_first(
        sim_state,
        &s1_move_choice,
        &s2_move_choice,
        &mut instructions,
    );

    // Determine who moves first (accounting for speed ties)
    let moves_first_side_one = match first_mover {
        SideMovesFirst::SideOne => true,
        SideMovesFirst::SideTwo => false,
        SideMovesFirst::SpeedTie => true, // In case of tie, we favor side one for win percentage calculation
    };

    // Get Pokémon stats
    let s1_pokemon = sim_state.side_one.get_active_immutable();
    let s2_pokemon = sim_state.side_two.get_active_immutable();

    // Calculate damage ranges
    let s1_damage_range = get_damage_range(sim_state, &s1_best_choice, &SideReference::SideOne);
    let s2_damage_range = get_damage_range(sim_state, &s2_best_choice, &SideReference::SideTwo);

    // Calculate recovery amounts (per turn)
    let s1_recovery_per_turn = calculate_recovery_per_turn(sim_state, &SideReference::SideOne);
    let s2_recovery_per_turn = calculate_recovery_per_turn(sim_state, &SideReference::SideTwo);

    // Calculate priority move damage
    let s1_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideOne);
    let s2_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideTwo);

    // Get average and max damage
    let (s1_avg_damage, s1_max_damage) = if s1_damage_range.is_empty() {
        (0, 0)
    } else {
        let avg =
            (s1_damage_range.iter().sum::<i16>() as f32 / s1_damage_range.len() as f32) as i16;
        let max = *s1_damage_range.iter().max().unwrap_or(&0);
        (avg, max)
    };

    let (s2_avg_damage, s2_max_damage) = if s2_damage_range.is_empty() {
        (0, 0)
    } else {
        let avg =
            (s2_damage_range.iter().sum::<i16>() as f32 / s2_damage_range.len() as f32) as i16;
        let max = *s2_damage_range.iter().max().unwrap_or(&0);
        (avg, max)
    };

    // Calculate turns to KO (accounting for recovery)
    let s1_turns_to_ko = calculate_turns_to_ko(s2_pokemon.hp, s1_avg_damage, s2_recovery_per_turn);
    let s2_turns_to_ko = calculate_turns_to_ko(s1_pokemon.hp, s2_avg_damage, s1_recovery_per_turn);

    // Check for OHKO potential
    let s1_ohko_chance = if s1_damage_range.is_empty() {
        0.0
    } else {
        let ohko_count = s1_damage_range
            .iter()
            .filter(|&&dmg| dmg >= s2_pokemon.hp)
            .count();
        ohko_count as f32 / s1_damage_range.len() as f32
    };

    let s2_ohko_chance = if s2_damage_range.is_empty() {
        0.0
    } else {
        let ohko_count = s2_damage_range
            .iter()
            .filter(|&&dmg| dmg >= s1_pokemon.hp)
            .count();
        ohko_count as f32 / s2_damage_range.len() as f32
    };

    // Check setup viability
    let s1_can_setup_safely = if s1_has_setup {
        let setup_turns_needed = 1; // Most setup moves need just 1 turn
        let damage_taken_during_setup = setup_turns_needed * s2_max_damage;
        s1_pokemon.hp > damage_taken_during_setup
    } else {
        false
    };

    let s2_can_setup_safely = if s2_has_setup {
        let setup_turns_needed = 1; // Most setup moves need just 1 turn
        let damage_taken_during_setup = setup_turns_needed * s1_max_damage;
        s2_pokemon.hp > damage_taken_during_setup
    } else {
        false
    };

    // Check post-setup sweep potential
    let s1_boosts_speed = has_speed_boosting_move(sim_state, &SideReference::SideOne);
    let s2_boosts_speed = has_speed_boosting_move(sim_state, &SideReference::SideTwo);

    let s1_boosted_damage = s1_avg_damage * 2; // Approximate 1 stage of Attack/SpAtk boost
    let s2_boosted_damage = s2_avg_damage * 2; // Approximate 1 stage of Attack/SpAtk boost

    let s1_setup_ohko = s1_boosted_damage >= s2_pokemon.hp;
    let s2_setup_ohko = s2_boosted_damage >= s1_pokemon.hp;

    // Recovery dominance check (immune to KO due to recovery)
    let s1_recovery_dominates = s1_recovery_per_turn > 2 * s2_avg_damage;
    let s2_recovery_dominates = s2_recovery_per_turn > 2 * s1_avg_damage;

    // Recovery sufficiency check (can partially offset damage)
    let s1_recovery_sufficient = s1_recovery_per_turn > (s2_avg_damage as f32 * 1.5) as i16;
    let s2_recovery_sufficient = s2_recovery_per_turn > (s1_avg_damage as f32 * 1.5) as i16;

    // Check for priority KO
    let s1_priority_ko = s1_priority_damage >= s2_pokemon.hp;
    let s2_priority_ko = s2_priority_damage >= s1_pokemon.hp;

    // Now apply our mathematical classification framework

    // COUNTER CONDITIONS (90%+ win rate)
    if s1_recovery_dominates {
        // Recovery exceeds all possible damage from opponent
        return 0.95;
    }

    if s1_priority_ko {
        // Can KO with priority before opponent moves
        return 0.95;
    }

    if moves_first_side_one && s1_ohko_chance > 0.9 {
        // Has nearly guaranteed OHKO and moves first
        return 0.95;
    }

    if moves_first_side_one && s1_turns_to_ko <= 2 && s2_turns_to_ko >= 3 {
        // Much faster KO and moves first
        return 0.95;
    }

    if s1_can_setup_safely && (s1_boosts_speed || moves_first_side_one) && s1_setup_ohko {
        // Can safely setup and then OHKO
        return 0.9;
    }

    // CHECK CONDITIONS (70-90% win rate)
    if s1_recovery_sufficient && !s2_recovery_sufficient {
        // Recovery offsets most but not all damage
        return 0.8;
    }

    if moves_first_side_one && s1_turns_to_ko <= s2_turns_to_ko {
        // Faster and equal/better KO speed
        return 0.75;
    }

    if s1_can_setup_safely && s1_setup_ohko && !moves_first_side_one {
        // Can setup but sweeping isn't guaranteed
        return 0.7;
    }

    // NEUTRAL CONDITIONS (30-70% win rate)
    if s1_turns_to_ko == s2_turns_to_ko {
        // Equal KO timing - speed determines winner
        return if moves_first_side_one { 0.6 } else { 0.4 };
    }

    if (moves_first_side_one && s1_turns_to_ko > s2_turns_to_ko)
        || (!moves_first_side_one && s1_turns_to_ko < s2_turns_to_ko)
    {
        // Offsetting advantages
        return 0.5;
    }

    // COUNTERED CONDITIONS (<10% win rate)
    if s2_recovery_dominates {
        // Opponent can recover more than max damage output
        return 0.05;
    }

    if s2_priority_ko {
        // Opponent KOs with priority before can move
        return 0.05;
    }

    if !moves_first_side_one && s2_ohko_chance > 0.9 {
        // Opponent almost guaranteed to OHKO and moves first
        return 0.05;
    }

    if !moves_first_side_one && s2_turns_to_ko <= 2 && s1_turns_to_ko >= 3 {
        // Opponent has much faster KO and moves first
        return 0.05;
    }

    if s2_can_setup_safely && (s2_boosts_speed || !moves_first_side_one) && s2_setup_ohko {
        // Opponent can safely set up and OHKO
        return 0.1;
    }

    // CHECKED CONDITIONS (10-30% win rate)
    if !moves_first_side_one && s1_turns_to_ko == s2_turns_to_ko {
        // Equal KO timing but moves second
        return 0.25;
    }

    if s1_turns_to_ko > s2_turns_to_ko && !s1_recovery_sufficient {
        // Takes longer to KO and recovery isn't sufficient
        return 0.2;
    }

    // Default - neutral matchup
    return 0.5;
}

/// Calculate turns needed to KO, accounting for recovery
pub fn calculate_turns_to_ko(target_hp: i16, damage_per_turn: i16, recovery_per_turn: i16) -> i32 {
    if damage_per_turn <= 0 {
        return 99; // Can't KO
    }

    if damage_per_turn <= recovery_per_turn {
        return 16; // Recovery outpaces damage
    }

    // Effective damage = damage - recovery
    let effective_damage = damage_per_turn - recovery_per_turn;

    // Calculate turns to KO
    ((target_hp as f32) / (effective_damage as f32)).ceil() as i32
}

/// Calculate recovery per turn from recovery moves
pub fn calculate_recovery_per_turn(state: &State, side_ref: &SideReference) -> i16 {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();
    let mut result = 0;

    // Check for recovery moves
    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        match m.id {
            // 50% recovery moves
            Choices::RECOVER
            | Choices::ROOST
            | Choices::MOONLIGHT
            | Choices::MORNINGSUN
            | Choices::SYNTHESIS
            | Choices::SLACKOFF
            | Choices::MILKDRINK
            | Choices::SOFTBOILED
            | Choices::WISH
            | Choices::HEALORDER => {
                result = max(result, pokemon.maxhp / 2);
            }

            // 25% recovery moves
            Choices::AQUARING | Choices::INGRAIN | Choices::REST => {
                result = max(result, pokemon.maxhp / 4);
            }

            // Draining moves (estimated average recovery)
            Choices::DRAINPUNCH
            | Choices::GIGADRAIN
            | Choices::HORNLEECH
            | Choices::DRAININGKISS
            | Choices::OBLIVIONWING
            | Choices::PARABOLICCHARGE => {
                // Estimate recovery at 20% of max HP (assuming average damage)
                result = max(result, pokemon.maxhp / 6);
            }

            // Leech Seed recovery
            Choices::LEECHSEED => {
                result = max(result, pokemon.maxhp / 8);
            }

            // Item recovery
            _ => {}
        }
    }

    // Check for Leftovers recovery
    if pokemon.item == Items::LEFTOVERS {
        result += pokemon.maxhp / 16;
    }

    result
}

/// Get the best priority move damage for a Pokémon
pub fn get_best_priority_move_damage(state: &State, side_ref: &SideReference) -> i16 {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();
    let mut best_priority_damage = 0;

    // Check for priority moves
    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        if m.choice.priority > 0 && m.choice.category != MoveCategory::Status {
            // Calculate damage for this priority move
            if let Some(damage_range) =
                calculate_damage_rolls(state.clone(), side_ref, m.choice.clone(), &m.choice)
            {
                if !damage_range.is_empty() {
                    // Use average damage as the metric
                    let avg_damage = (damage_range.iter().sum::<i16>() as f32
                        / damage_range.len() as f32) as i16;
                    if avg_damage > best_priority_damage {
                        best_priority_damage = avg_damage;
                    }
                }
            }
        }
    }

    best_priority_damage
}

/// Find the best move for a given Pokémon
pub fn find_best_move(state: &State, side_ref: &SideReference) -> (Choice, PokemonMoveIndex) {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();
    let mut best_choice = Choice::default();
    let mut best_damage = 0;
    let mut best_move_index = PokemonMoveIndex::M0;

    // Analyze each move
    let mut iter = pokemon.moves.into_iter();
    while let Some(m) = iter.next() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        let mut choice = m.choice.clone();
        choice.move_index = iter.pokemon_move_index;

        // Skip moves that would have no effect or can't be used
        if move_has_no_effect(state, &choice, side_ref) || cannot_use_move(state, &choice, side_ref)
        {
            continue;
        }

        // Only consider damaging moves
        if choice.category != MoveCategory::Status {
            let damage_rolls =
                calculate_damage_rolls(state.clone(), side_ref, choice.clone(), &choice);

            if let Some(damage_range) = damage_rolls {
                if !damage_range.is_empty() {
                    // Calculate average damage
                    let avg_damage =
                        damage_range.iter().sum::<i16>() as f32 / damage_range.len() as f32;

                    // Check if this move is better than current best
                    if avg_damage as i16 > best_damage {
                        best_damage = avg_damage as i16;
                        best_choice = choice;
                        best_move_index = iter.pokemon_move_index;
                    }
                }
            }
        }
    }

    (best_choice, best_move_index)
}

/// Get damage range for a specific move
pub fn get_damage_range(state: &State, choice: &Choice, side_ref: &SideReference) -> Vec<i16> {
    if choice.category == MoveCategory::Status {
        return Vec::new();
    }

    // Use existing damage calculation function
    match calculate_damage_rolls(state.clone(), side_ref, choice.clone(), choice) {
        Some(damage_range) => damage_range,
        None => Vec::new(),
    }
}

/// Check if a move is a setup move
fn is_setup_move(move_id: &Choices) -> bool {
    matches!(
        move_id,
        Choices::SWORDSDANCE
            | Choices::NASTYPLOT
            | Choices::DRAGONDANCE
            | Choices::CALMMIND
            | Choices::BULKUP
            | Choices::SHELLSMASH
            | Choices::QUIVERDANCE
            | Choices::CLANGOROUSSOUL
            | Choices::VICTORYDANCE
            | Choices::CURSE
            | Choices::TAILGLOW
            | Choices::AGILITY
            | Choices::AUTOTOMIZE
            | Choices::GROWTH
            | Choices::WORKUP
            | Choices::COIL
    )
}

/// Determine if a Pokémon has a setup move
pub fn has_setup_move(state: &State, side_ref: &SideReference) -> bool {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();

    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        if is_setup_move(&m.id) {
            return true;
        }
    }

    false
}

/// Check if Pokémon has a speed-boosting move
pub fn has_speed_boosting_move(state: &State, side_ref: &SideReference) -> bool {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();

    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        match m.id {
            // Speed-boosting moves
            Choices::DRAGONDANCE
            | Choices::SHELLSMASH
            | Choices::QUIVERDANCE
            | Choices::AGILITY
            | Choices::AUTOTOMIZE
            | Choices::ROCKPOLISH
            | Choices::VICTORYDANCE => {
                return true;
            }
            _ => {}
        }
    }

    false
}

/// Determine if a Pokémon has status-inducing moves
fn has_status_move(state: &State, side_ref: &SideReference) -> bool {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();

    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        if is_status_move(&m.id) {
            return true;
        }
    }

    false
}

/// Check if a move induces status
fn is_status_move(move_id: &Choices) -> bool {
    matches!(
        move_id,
        Choices::TOXIC
            | Choices::THUNDERWAVE
            | Choices::WILLOWISP
            | Choices::HYPNOSIS
            | Choices::SLEEPPOWDER
            | Choices::STUNSPORE
            | Choices::POISONPOWDER
            | Choices::GLARE
            | Choices::DARKVOID
            | Choices::SPORE
            | Choices::YAWN
    ) || has_high_status_chance(move_id)
}

/// Check if a move has a high chance of causing status
fn has_high_status_chance(move_id: &Choices) -> bool {
    matches!(
        move_id,
        Choices::BODYSLAM | Choices::DISCHARGE | Choices::NUZZLE |  // Paralysis
        Choices::SLUDGEBOMB | Choices::GUNKSHOT | Choices::POISONJAB | // Poison
        Choices::SCALD | Choices::STEAMERUPTION | Choices::BURNINGJEALOUSY | // Burn
        Choices::FREEZESHOCK | Choices::ICEFANG // Freeze
    )
}

/// Classify matchup based on win percentage
pub fn classify_matchup_result(win_percentage: f32) -> i8 {
    if win_percentage > STRONG_FAVORABLE_THRESHOLD {
        // Strongly favorable (counter)
        2
    } else if win_percentage > FAVORABLE_THRESHOLD {
        // Favorable (check)
        1
    } else if win_percentage < STRONG_UNFAVORABLE_THRESHOLD {
        // Strongly unfavorable (hard countered)
        -2
    } else if win_percentage < UNFAVORABLE_THRESHOLD {
        // Unfavorable (checked)
        -1
    } else {
        // Neutral matchup
        0
    }
}

/// Analyzes the matchup between the two Pokémon using advanced mathematical win conditions
fn analyze_win_percentage_enhanced(sim_state: &mut State) -> f32 {
    // Find the best move choices for each Pokémon
    let (s1_best_choice, s1_best_index) = find_best_move(sim_state, &SideReference::SideOne);
    let (s2_best_choice, s2_best_index) = find_best_move(sim_state, &SideReference::SideTwo);

    // Get Pokémon stats and info
    let s1_pokemon = sim_state.side_one.get_active_immutable();
    let s2_pokemon = sim_state.side_two.get_active_immutable();

    // Analyze speed to determine who goes first
    let mut s1_move_choice = s1_best_choice.clone();
    let mut s2_move_choice = s2_best_choice.clone();
    s1_move_choice.first_move = true;
    s2_move_choice.first_move = true;

    let mut instructions = StateInstructions::default();
    let first_mover = moves_first(
        sim_state,
        &s1_move_choice,
        &s2_move_choice,
        &mut instructions,
    );

    // Determine who moves first (accounting for speed ties)
    let moves_first_side_one = match first_mover {
        SideMovesFirst::SideOne => true,
        SideMovesFirst::SideTwo => false,
        SideMovesFirst::SpeedTie => true, // In case of tie, we favor side one for win percentage calculation
    };

    // Calculate damage ranges
    let s1_damage_range = get_damage_range(sim_state, &s1_best_choice, &SideReference::SideOne);
    let s2_damage_range = get_damage_range(sim_state, &s2_best_choice, &SideReference::SideTwo);

    // Get recovery capabilities
    let s1_recovery_info = analyze_recovery_capabilities(sim_state, &SideReference::SideOne);
    let s2_recovery_info = analyze_recovery_capabilities(sim_state, &SideReference::SideTwo);

    // Get setup capabilities
    let s1_setup_info = analyze_setup_capabilities(sim_state, &SideReference::SideOne);
    let s2_setup_info = analyze_setup_capabilities(sim_state, &SideReference::SideTwo);

    // Get average and max damage
    let (s1_avg_damage, s1_max_damage) = if s1_damage_range.is_empty() {
        (0, 0)
    } else {
        let avg =
            (s1_damage_range.iter().sum::<i16>() as f32 / s1_damage_range.len() as f32) as i16;
        let max = *s1_damage_range.iter().max().unwrap_or(&0);
        (avg, max)
    };

    let (s2_avg_damage, s2_max_damage) = if s2_damage_range.is_empty() {
        (0, 0)
    } else {
        let avg =
            (s2_damage_range.iter().sum::<i16>() as f32 / s2_damage_range.len() as f32) as i16;
        let max = *s2_damage_range.iter().max().unwrap_or(&0);
        (avg, max)
    };

    // Calculate priority move damage
    let s1_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideOne);
    let s2_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideTwo);

    // ======= Recovery-based win condition analysis =======

    // Calculate optimal strategy parameters
    let s1_recovery_strategy = calculate_optimal_recovery_strategy(
        s1_pokemon.hp,
        s1_pokemon.maxhp,
        s2_avg_damage,
        s1_avg_damage,
        s1_recovery_info.recovery_amount,
        s1_recovery_info.recovery_pp,
        moves_first_side_one,
        s2_pokemon.hp,
    );

    let s2_recovery_strategy = calculate_optimal_recovery_strategy(
        s2_pokemon.hp,
        s2_pokemon.maxhp,
        s1_avg_damage,
        s2_avg_damage,
        s2_recovery_info.recovery_amount,
        s2_recovery_info.recovery_pp,
        !moves_first_side_one,
        s1_pokemon.hp,
    );

    // ======= Setup-based win condition analysis =======

    // Calculate whether setup is viable
    let s1_setup_viable = is_setup_viable(
        s1_pokemon.hp,
        s2_avg_damage,
        s2_max_damage,
        s1_setup_info.turns_needed,
        s1_recovery_info.has_recovery,
        s1_recovery_info.recovery_amount,
        moves_first_side_one,
    );

    let s2_setup_viable = is_setup_viable(
        s2_pokemon.hp,
        s1_avg_damage,
        s1_max_damage,
        s2_setup_info.turns_needed,
        s2_recovery_info.has_recovery,
        s2_recovery_info.recovery_amount,
        !moves_first_side_one,
    );

    // Calculate post-setup damage scaling
    let s1_boosted_damage = calculate_boosted_damage(
        s1_avg_damage,
        s1_setup_info.attack_stages,
        s1_setup_info.special_attack_stages,
    );

    let s2_boosted_damage = calculate_boosted_damage(
        s2_avg_damage,
        s2_setup_info.attack_stages,
        s2_setup_info.special_attack_stages,
    );

    // ======= Win condition calculations with both setup and recovery =======

    // 1. Base turns to KO (without setup or recovery)
    let base_s1_turns_to_ko = (s2_pokemon.hp as f32 / s1_avg_damage as f32).ceil() as i32;
    let base_s2_turns_to_ko = (s1_pokemon.hp as f32 / s2_avg_damage as f32).ceil() as i32;

    // 2. Turns to KO with recovery
    let s1_turns_to_ko_with_recovery = calculate_turns_to_ko_optimized(
        s2_pokemon.hp,
        s1_avg_damage,
        s2_recovery_info.recovery_amount,
        s2_recovery_info.recovery_frequency,
        s2_recovery_info.recovery_pp,
    );

    let s2_turns_to_ko_with_recovery = calculate_turns_to_ko_optimized(
        s1_pokemon.hp,
        s2_avg_damage,
        s1_recovery_info.recovery_amount,
        s1_recovery_info.recovery_frequency,
        s1_recovery_info.recovery_pp,
    );

    // 3. Turns to KO with setup
    let s1_turns_to_ko_with_setup = if s1_setup_viable {
        s1_setup_info.turns_needed + (s2_pokemon.hp as f32 / s1_boosted_damage as f32).ceil() as i32
    } else {
        99 // Not viable
    };

    let s2_turns_to_ko_with_setup = if s2_setup_viable {
        s2_setup_info.turns_needed + (s1_pokemon.hp as f32 / s2_boosted_damage as f32).ceil() as i32
    } else {
        99 // Not viable
    };

    // 4. Turns to KO with both setup and recovery
    let s1_turns_to_ko_optimized = s1_turns_to_ko_with_setup
        .min(s1_turns_to_ko_with_recovery)
        .min(base_s1_turns_to_ko);
    let s2_turns_to_ko_optimized = s2_turns_to_ko_with_setup
        .min(s2_turns_to_ko_with_recovery)
        .min(base_s2_turns_to_ko);

    // 5. Recovery sustainability
    let s1_recovery_sustains = s1_recovery_info.recovery_amount > s2_avg_damage / 2;
    let s2_recovery_sustains = s2_recovery_info.recovery_amount > s1_avg_damage / 2;

    let s1_recovery_outlasts = s1_recovery_strategy.sustainable
        && (s1_recovery_info.recovery_pp >= s1_recovery_strategy.recovery_uses_needed as i8);

    let s2_recovery_outlasts = s2_recovery_strategy.sustainable
        && (s2_recovery_info.recovery_pp >= s2_recovery_strategy.recovery_uses_needed as i8);

    // Priority KO check
    let s1_priority_ko = s1_priority_damage >= s2_pokemon.hp;
    let s2_priority_ko = s2_priority_damage >= s1_pokemon.hp;

    // ======= Overall win probability calculation =======

    // HARD COUNTER CONDITIONS (95% win rate)
    if s1_priority_ko && !s2_priority_ko {
        // Can KO with priority while opponent cannot
        return 0.95;
    }

    if s1_recovery_strategy.recovery_dominates && !s2_recovery_strategy.recovery_dominates {
        // Recovery completely negates damage and opponent's doesn't
        return 0.95;
    }

    if moves_first_side_one && s1_avg_damage >= s2_pokemon.hp && s2_avg_damage < s1_pokemon.hp {
        // One-shot KO while moving first and not being one-shot back
        return 0.95;
    }

    // STRONG COUNTER CONDITIONS (90% win rate)
    if s1_setup_viable && !s2_setup_viable && s1_boosted_damage >= s2_pokemon.hp {
        // Can set up safely and then OHKO while opponent cannot
        return 0.9;
    }

    if s1_recovery_sustains && !s2_recovery_sustains && s1_turns_to_ko_optimized < 16 {
        // Recovery sustains and can eventually KO while opponent cannot sustain
        return 0.9;
    }

    if s1_turns_to_ko_optimized <= 3 && s2_turns_to_ko_optimized >= 6 {
        // Much faster KO time even accounting for recovery/setup
        return 0.9;
    }

    // FAVORABLE CONDITIONS (80% win rate)
    if moves_first_side_one && s1_turns_to_ko_optimized < s2_turns_to_ko_optimized {
        // Faster and better KO timing with optimal strategy
        return 0.8;
    }

    if s1_recovery_outlasts && !s2_recovery_outlasts {
        // Recovery strategy is sustainable while opponent's isn't
        return 0.8;
    }

    // SLIGHTLY FAVORABLE (70% win rate)
    if s1_setup_viable && s1_boosted_damage as f32 > 1.5 * s2_avg_damage as f32 {
        // Can set up and then outpace opponent
        return 0.7;
    }

    if s1_turns_to_ko_optimized < s2_turns_to_ko_optimized {
        // Simply faster KO with optimal strategy
        return 0.7;
    }

    // NEUTRAL CONDITIONS (40-60% win rate)
    if s1_turns_to_ko_optimized == s2_turns_to_ko_optimized {
        // Equal KO timing - speed advantage determines winner
        return if moves_first_side_one { 0.6 } else { 0.4 };
    }

    if (s1_recovery_sustains && s2_recovery_sustains)
        || (s1_setup_viable
            && s2_setup_viable
            && (s1_boosted_damage > s2_boosted_damage) == moves_first_side_one)
    {
        // Both have similar strategic advantages
        return 0.5;
    }

    // UNFAVORABLE CONDITIONS (30% win rate)
    if s2_turns_to_ko_optimized < s1_turns_to_ko_optimized {
        // Opponent has faster KO with optimal strategy
        return 0.3;
    }

    if s2_setup_viable && s2_boosted_damage as f32 > 1.5 * s1_avg_damage as f32 {
        // Opponent can set up and outpace
        return 0.3;
    }

    // STRONGLY UNFAVORABLE (20% win rate)
    if s2_recovery_outlasts && !s1_recovery_outlasts {
        // Opponent's recovery strategy is sustainable while ours isn't
        return 0.2;
    }

    if !moves_first_side_one && s2_turns_to_ko_optimized < s1_turns_to_ko_optimized {
        // Opponent is faster and has better KO timing
        return 0.2;
    }

    // HARD COUNTERED (5-10% win rate)
    if s2_priority_ko && !s1_priority_ko {
        // Opponent can KO with priority while we cannot
        return 0.05;
    }

    if s2_recovery_strategy.recovery_dominates && !s1_recovery_strategy.recovery_dominates {
        // Opponent's recovery completely negates damage and ours doesn't
        return 0.05;
    }

    if !moves_first_side_one && s2_avg_damage >= s1_pokemon.hp && s1_avg_damage < s2_pokemon.hp {
        // Opponent can one-shot KO while moving first and not being one-shot back
        return 0.05;
    }

    if s2_setup_viable && !s1_setup_viable && s2_boosted_damage >= s1_pokemon.hp {
        // Opponent can set up safely and then OHKO while we cannot
        return 0.1;
    }

    // Default neutral
    return 0.5;
}

/// Recovery strategy analysis results
struct RecoveryStrategy {
    sustainable: bool,          // Is the recovery strategy sustainable?
    recovery_dominates: bool,   // Does recovery completely negate damage?
    optimal_threshold: i16,     // HP threshold where recovery becomes optimal
    recovery_frequency: f32,    // Optimal frequency of recovery (fraction of turns)
    recovery_uses_needed: i32,  // Total recovery uses needed for the battle
    expected_battle_turns: i32, // Expected battle duration in turns
}

/// Calculate the optimal recovery strategy for a Pokémon
fn calculate_optimal_recovery_strategy(
    current_hp: i16,
    max_hp: i16,
    incoming_damage: i16,
    outgoing_damage: i16,
    recovery_amount: i16,
    recovery_pp: i8,
    moves_first: bool,
    opponent_hp: i16,
) -> RecoveryStrategy {
    // If no damage or recovery, return default strategy
    if incoming_damage == 0 || recovery_amount == 0 {
        return RecoveryStrategy {
            sustainable: false,
            recovery_dominates: false,
            optimal_threshold: 0,
            recovery_frequency: 0.0,
            recovery_uses_needed: 0,
            expected_battle_turns: if outgoing_damage > 0 {
                (opponent_hp as f32 / outgoing_damage as f32).ceil() as i32
            } else {
                99
            },
        };
    }

    // Calculate whether recovery dominates damage
    let recovery_dominates = if moves_first {
        // If moving first, recovery dominates if it exceeds incoming damage
        recovery_amount > incoming_damage
    } else {
        // If moving second, recovery dominates if it exceeds twice the incoming damage
        // (need to recover from two hits: one before recovery, one after)
        recovery_amount > 2 * incoming_damage
    };

    // Calculate optimal HP threshold for recovery
    let optimal_threshold = if moves_first {
        // When faster, recover if HP would drop below incoming damage after attacking
        incoming_damage
    } else {
        // When slower, recover if current HP is not enough to survive a hit plus maintain threshold
        2 * incoming_damage
    };

    // Calculate optimal recovery frequency (ratio of recovery turns to total turns)
    let damage_per_turn = incoming_damage as f32;
    let recovery_per_use = recovery_amount as f32;

    // For sustainable recovery, need: recovery_per_use ÷ frequency ≥ damage_per_turn
    // Therefore: frequency ≥ recovery_per_use ÷ damage_per_turn
    let theoretical_frequency = if damage_per_turn > 0.0 {
        recovery_per_use / damage_per_turn
    } else {
        0.0 // No incoming damage
    };

    // Clamp to realistic values: frequency must be between 0 and 1
    let recovery_frequency = theoretical_frequency.min(1.0).max(0.0);

    // A strategy is sustainable if theoretical frequency <= 1.0
    // This means recovery can at least keep pace with incoming damage
    let sustainable = theoretical_frequency <= 1.0 && theoretical_frequency > 0.0;

    // Calculate expected battle duration
    let expected_turns_to_ko = if outgoing_damage > 0 {
        // Effective outgoing damage is reduced by recovery frequency
        let effective_outgoing_damage = outgoing_damage as f32 * (1.0 - recovery_frequency);
        if effective_outgoing_damage > 0.0 {
            (opponent_hp as f32 / effective_outgoing_damage).ceil() as i32
        } else {
            99 // Can't KO if always recovering
        }
    } else {
        99 // Can't deal damage
    };

    // Calculate total recovery uses needed for the battle
    let recovery_uses_needed = (expected_turns_to_ko as f32 * recovery_frequency).ceil() as i32;

    RecoveryStrategy {
        sustainable,
        recovery_dominates,
        optimal_threshold,
        recovery_frequency,
        recovery_uses_needed,
        expected_battle_turns: expected_turns_to_ko,
    }
}

/// Enhanced turns to KO calculation that accounts for optimized recovery usage
fn calculate_turns_to_ko_optimized(
    target_hp: i16,
    damage_per_turn: i16,
    recovery_per_use: i16,
    recovery_frequency: f32,
    recovery_pp: i8,
) -> i32 {
    if damage_per_turn <= 0 {
        return 99; // Can't KO
    }

    // Effective damage accounting for recovery
    let effective_damage_per_turn = damage_per_turn as f32 * (1.0 - recovery_frequency)
        - (recovery_per_use as f32 * recovery_frequency);

    if effective_damage_per_turn <= 0.0 {
        // Recovery outpaces damage, but check if PP is sufficient
        let theoretical_turns = 99;
        let recovery_uses_needed = (theoretical_turns as f32 * recovery_frequency).ceil() as i32;

        if recovery_uses_needed <= recovery_pp as i32 {
            return theoretical_turns; // Can stall indefinitely
        } else {
            // Calculate how long recovery PP will last
            let recovery_turns = (recovery_pp as f32 / recovery_frequency).floor() as i32;

            // After recovery PP is exhausted, how many more turns to KO?
            let remaining_hp_after_recovery =
                target_hp as i32 - (damage_per_turn as i32 * recovery_turns);

            if remaining_hp_after_recovery <= 0 {
                return recovery_turns; // KO'd before recovery PP runs out
            } else {
                // Additional turns needed after recovery PP is exhausted
                let additional_turns =
                    (remaining_hp_after_recovery as f32 / damage_per_turn as f32).ceil() as i32;
                return recovery_turns + additional_turns;
            }
        }
    } else {
        // Recovery doesn't fully offset damage
        let theoretical_turns = (target_hp as f32 / effective_damage_per_turn).ceil() as i32;

        // Check if recovery PP is sufficient for the calculated duration
        let recovery_uses_needed = (theoretical_turns as f32 * recovery_frequency).ceil() as i32;

        if recovery_uses_needed <= recovery_pp as i32 {
            return theoretical_turns;
        } else {
            // Calculate adjusted turns with limited recovery PP
            let recovery_turns = (recovery_pp as f32 / recovery_frequency).floor() as i32;
            let damage_during_recovery = (damage_per_turn as f32 * recovery_turns as f32)
                - (recovery_per_use as f32 * recovery_pp as f32);

            let remaining_hp = target_hp as f32 - damage_during_recovery;
            if remaining_hp <= 0.0 {
                return recovery_turns;
            } else {
                let additional_turns = (remaining_hp / damage_per_turn as f32).ceil() as i32;
                return recovery_turns + additional_turns;
            }
        }
    }
}

/// Recovery capabilities for a Pokémon
struct RecoveryInfo {
    has_recovery: bool,      // Does the Pokémon have recovery moves?
    recovery_amount: i16,    // HP recovered per use
    recovery_pp: i8,         // Total PP of the recovery move
    recovery_frequency: f32, // Optimal frequency for recovery use (fraction of turns)
    drain_multiplier: f32,   // For drain moves, multiplier of damage dealt
}

/// Analyze recovery capabilities of a Pokémon
fn analyze_recovery_capabilities(state: &State, side_ref: &SideReference) -> RecoveryInfo {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();
    let mut has_recovery = false;
    let mut recovery_amount = 0i16;
    let mut recovery_pp = 0i8;
    let mut drain_multiplier = 0.0f32;

    // Check for recovery moves
    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        match m.id {
            // 50% recovery moves
            Choices::RECOVER
            | Choices::ROOST
            | Choices::MOONLIGHT
            | Choices::MORNINGSUN
            | Choices::SYNTHESIS
            | Choices::SLACKOFF
            | Choices::MILKDRINK
            | Choices::SOFTBOILED
            | Choices::WISH
            | Choices::HEALORDER => {
                has_recovery = true;
                let move_recovery = pokemon.maxhp / 2;
                if move_recovery > recovery_amount {
                    recovery_amount = move_recovery;
                    recovery_pp = m.pp;
                }
            }

            // 25% recovery moves
            Choices::AQUARING | Choices::INGRAIN | Choices::JUNGLEHEALING => {
                has_recovery = true;
                let move_recovery = pokemon.maxhp / 4;
                if move_recovery > recovery_amount {
                    recovery_amount = move_recovery;
                    recovery_pp = m.pp;
                }
            }

            // Drain moves
            Choices::DRAINPUNCH
            | Choices::GIGADRAIN
            | Choices::HORNLEECH
            | Choices::DRAININGKISS
            | Choices::OBLIVIONWING
            | Choices::PARABOLICCHARGE => {
                has_recovery = true;
                drain_multiplier = 0.5; // 50% of damage dealt
                                        // Note: We don't set recovery_amount for drain moves since it varies
            }

            // 33.3% Recovery moves
            Choices::REST => {
                has_recovery = true;
                let move_recovery = pokemon.maxhp;
                if move_recovery > recovery_amount {
                    recovery_amount = move_recovery;
                    recovery_pp = m.pp;
                }
            }

            // Leech Seed and other passive recovery
            Choices::LEECHSEED => {
                has_recovery = true;
                let move_recovery = pokemon.maxhp / 8;
                if move_recovery > recovery_amount {
                    recovery_amount = move_recovery;
                    recovery_pp = m.pp;
                }
            }

            _ => {}
        }
    }

    // Check for Leftovers or other passive recovery
    if pokemon.item == Items::LEFTOVERS || pokemon.item == Items::BLACKSLUDGE {
        has_recovery = true;
        let item_recovery = pokemon.maxhp / 16;
        if item_recovery > recovery_amount {
            recovery_amount = item_recovery;
            recovery_pp = 16; // Arbitrary high value for passive recovery
        }
    }

    // Calculate optimal recovery frequency
    let recovery_frequency = if recovery_amount > 0 { 0.5 } else { 0.0 };

    RecoveryInfo {
        has_recovery,
        recovery_amount,
        recovery_pp,
        recovery_frequency,
        drain_multiplier,
    }
}

/// Setup capabilities for a Pokémon
struct SetupInfo {
    has_setup: bool,             // Does the Pokémon have setup moves?
    turns_needed: i32,           // Number of turns needed for setup
    attack_stages: i32,          // Attack boost stages
    defense_stages: i32,         // Defense boost stages
    special_attack_stages: i32,  // Special Attack boost stages
    special_defense_stages: i32, // Special Defense boost stages
    speed_stages: i32,           // Speed boost stages
    boosts_both_offenses: bool,  // Does the setup boost both physical and special?
    boosts_both_defenses: bool,  // Does the setup boost both physical and special defenses?
    boosts_speed: bool,          // Does the setup boost speed?
}

/// Analyze setup capabilities of a Pokémon
fn analyze_setup_capabilities(state: &State, side_ref: &SideReference) -> SetupInfo {
    let side = match side_ref {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };

    let pokemon = side.get_active_immutable();
    let mut has_setup = false;
    let mut attack_stages = 0;
    let mut defense_stages = 0;
    let mut special_attack_stages = 0;
    let mut special_defense_stages = 0;
    let mut speed_stages = 0;
    let mut boosts_both_offenses = false;
    let mut boosts_both_defenses = false;
    let mut boosts_speed = false;

    // Check for setup moves
    for m in pokemon.moves.into_iter() {
        if m.id == Choices::NONE || m.pp <= 0 || m.disabled {
            continue;
        }

        match m.id {
            // Attack boosts
            Choices::SWORDSDANCE => {
                has_setup = true;
                attack_stages += 2;
            }

            // Special Attack boosts
            Choices::NASTYPLOT | Choices::TAILGLOW => {
                has_setup = true;
                special_attack_stages += 2;
            }

            // Mixed setup moves
            Choices::DRAGONDANCE => {
                has_setup = true;
                attack_stages += 1;
                speed_stages += 1;
                boosts_speed = true;
            }
            Choices::CALMMIND => {
                has_setup = true;
                special_attack_stages += 1;
                special_defense_stages += 1;
                boosts_both_defenses = true;
            }
            Choices::BULKUP => {
                has_setup = true;
                attack_stages += 1;
                defense_stages += 1;
            }
            Choices::SHELLSMASH => {
                has_setup = true;
                attack_stages += 2;
                special_attack_stages += 2;
                speed_stages += 2;
                defense_stages -= 1;
                special_defense_stages -= 1;
                boosts_both_offenses = true;
                boosts_speed = true;
            }
            Choices::QUIVERDANCE => {
                has_setup = true;
                special_attack_stages += 1;
                special_defense_stages += 1;
                speed_stages += 1;
                boosts_speed = true;
            }
            Choices::CLANGOROUSSOUL => {
                has_setup = true;
                attack_stages += 1;
                defense_stages += 1;
                special_attack_stages += 1;
                special_defense_stages += 1;
                speed_stages += 1;
                boosts_both_offenses = true;
                boosts_both_defenses = true;
                boosts_speed = true;
            }
            Choices::VICTORYDANCE => {
                has_setup = true;
                attack_stages += 1;
                defense_stages += 1;
                speed_stages += 1;
                boosts_speed = true;
            }
            Choices::CURSE => {
                has_setup = true;
                attack_stages += 1;
                defense_stages += 1;
                speed_stages -= 1;
            }
            // Speed boosting moves
            Choices::AGILITY | Choices::AUTOTOMIZE | Choices::ROCKPOLISH => {
                has_setup = true;
                speed_stages += 2;
                boosts_speed = true;
            }
            // Growth and Work Up boost both offenses
            Choices::GROWTH | Choices::WORKUP => {
                has_setup = true;
                attack_stages += 1;
                special_attack_stages += 1;
                boosts_both_offenses = true;
            }
            // Coil boosts accuracy too, but we ignore that
            Choices::COIL => {
                has_setup = true;
                attack_stages += 1;
                defense_stages += 1;
            }
            // Cosmic Power/Defend Order boost both defenses
            Choices::COSMICPOWER | Choices::DEFENDORDER => {
                has_setup = true;
                defense_stages += 1;
                special_defense_stages += 1;
                boosts_both_defenses = true;
            }
            // Other stat-boosting moves
            Choices::IRONDEFENSE | Choices::COTTONGUARD | Choices::ACIDARMOR => {
                has_setup = true;
                defense_stages += 2;
            }
            Choices::AMNESIA => {
                has_setup = true;
                special_defense_stages += 2;
            }
            Choices::GEOMANCY => {
                has_setup = true;
                special_attack_stages += 2;
                special_defense_stages += 2;
                speed_stages += 2;
                boosts_both_defenses = true;
                boosts_speed = true;
            }
            Choices::HOWL => {
                has_setup = true;
                attack_stages += 1;
            }
            _ => {}
        }
    }

    // Determine optimal number of setup turns
    // For most cases, 1 turn is enough, but for specific strategies we might want more
    let mut turns_needed = if has_setup { 1 } else { 0 };

    // For multiple-stage setups, we might need more turns
    // This is a mathematical optimization based on damage potential vs. setup investment
    if has_setup {
        // Check if the Pokémon would benefit from multiple turns of setup
        if (attack_stages >= 2 || special_attack_stages >= 2)
            && (defense_stages > 0 || special_defense_stages > 0)
        {
            // Pokémon has both offensive and defensive boosts - might benefit from 2 turns
            // This is particularly true for Calm Mind + Stored Power users or Bulk Up + physical attackers
            turns_needed = 2;
        } else if boosts_speed
            && (speed_stages < 2)
            && (attack_stages > 0 || special_attack_stages > 0)
        {
            // Speed boost is important, but not enough to outspeed everything yet
            // Might need a second turn to further boost speed
            turns_needed = 2;
        }
    }

    SetupInfo {
        has_setup,
        turns_needed,
        attack_stages,
        defense_stages,
        special_attack_stages,
        special_defense_stages,
        speed_stages,
        boosts_both_offenses,
        boosts_both_defenses,
        boosts_speed,
    }
}

/// Determine if setup is viable based on matchup conditions
fn is_setup_viable(
    current_hp: i16,
    avg_damage: i16,
    max_damage: i16,
    setup_turns: i32,
    has_recovery: bool,
    recovery_amount: i16,
    moves_first: bool,
) -> bool {
    if setup_turns == 0 {
        return false; // No setup moves
    }

    // If we move first, we take damage after setting up
    let damage_during_setup = if moves_first {
        avg_damage * setup_turns as i16
    } else {
        // If we move second, we take damage before and after setting up
        avg_damage * (setup_turns as i16 + 1)
    };

    // If we have recovery, we can potentially recover during setup
    if has_recovery && recovery_amount > avg_damage {
        // Recovery outpaces damage; setup is viable regardless of current HP
        return true;
    } else if has_recovery {
        // With recovery that doesn't outpace damage, we need to factor it in
        let effective_damage = damage_during_setup - recovery_amount;
        return current_hp > effective_damage;
    } else {
        // Without recovery, we simply need enough HP to survive the setup phase
        // We use a slightly higher threshold with max damage to be conservative
        let safety_threshold = (max_damage * setup_turns as i16) + (max_damage / 4);
        return current_hp > safety_threshold;
    }
}

/// Calculate damage scaling after stat boosts
fn calculate_boosted_damage(
    base_damage: i16,
    attack_stages: i32,
    special_attack_stages: i32,
) -> i16 {
    // Use the higher of the two boost stages
    let effective_stages = attack_stages.max(special_attack_stages);

    if effective_stages <= 0 {
        return base_damage;
    }

    // Apply the standard damage multiplier formula from Pokémon games
    // +1 stage = 1.5x, +2 stages = 2x, +3 stages = 2.5x, etc.
    let multiplier = match effective_stages {
        1 => 1.5,
        2 => 2.0,
        3 => 2.5,
        4 => 3.0,
        5 => 3.5,
        6 => 4.0,
        _ => 4.0, // Cap at +6 stages
    };

    (base_damage as f32 * multiplier) as i16
}

/// Enhanced function to calculate PP-limited turns to KO
pub fn calculate_pp_limited_turns_to_ko(
    target_hp: i16,
    damage_per_turn: i16,
    recovery_per_turn: i16,
    attack_pp: i16,
    recovery_pp: i16,
    optimal_recovery_frequency: f32,
) -> i32 {
    if damage_per_turn <= 0 {
        return 99; // Can't KO
    }

    // Effective damage accounting for recovery
    let net_damage_per_attack_turn = damage_per_turn - recovery_per_turn;

    if net_damage_per_attack_turn <= 0 {
        // Recovery outpaces damage, check if PP is enough to maintain
        // The battle will be decided by who runs out of PP first

        // Calculate total effective PP considering optimal strategy
        let attack_turns = attack_pp as f32;
        let recovery_turns = recovery_pp as f32;
        let total_sustainable_turns = attack_turns + recovery_turns;

        return total_sustainable_turns as i32;
    }

    // Calculate how many attack turns are needed for KO
    let attack_turns_for_ko = (target_hp as f32 / net_damage_per_attack_turn as f32).ceil() as i32;

    // Check if we have enough attack PP
    if attack_turns_for_ko <= attack_pp as i32 {
        // Calculate corresponding recovery turns needed based on frequency
        let recovery_turns_needed = (attack_turns_for_ko as f32 * optimal_recovery_frequency
            / (1.0 - optimal_recovery_frequency)) as i32;

        // Check if we have enough recovery PP
        if recovery_turns_needed <= recovery_pp as i32 {
            return attack_turns_for_ko + recovery_turns_needed;
        } else {
            // Not enough recovery PP, recalculate with what we have
            let actual_recovery_turns = recovery_pp as i32;
            let sustainable_attack_turns = (actual_recovery_turns as f32
                * (1.0 - optimal_recovery_frequency)
                / optimal_recovery_frequency) as i32;

            // Damage dealt during sustainable phase
            let damage_dealt_sustainable =
                sustainable_attack_turns as i16 * net_damage_per_attack_turn;

            // Remaining HP after sustainable phase
            let remaining_hp = target_hp - damage_dealt_sustainable;
            if remaining_hp <= 0 {
                // KO achieved during sustainable phase
                return sustainable_attack_turns + actual_recovery_turns;
            } else {
                // Need additional attacks without recovery
                let additional_turns = (remaining_hp as f32 / damage_per_turn as f32).ceil() as i32;
                return sustainable_attack_turns + actual_recovery_turns + additional_turns;
            }
        }
    } else {
        // Not enough attack PP to KO
        return 99;
    }
}
