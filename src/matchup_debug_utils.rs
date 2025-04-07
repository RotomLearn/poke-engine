use crate::choices::{Choice, Choices};
use crate::generate_instructions::{calculate_damage_rolls, moves_first};
use crate::instruction::StateInstructions;
use crate::matchup_calc::classify_matchup_result;
use crate::matchup_calc::{
    calculate_recovery_per_turn, calculate_turns_to_ko, find_best_move,
    get_best_priority_move_damage, get_damage_range, has_setup_move, has_speed_boosting_move,
};
use crate::matchup_mcts::create_simulation_state;
use crate::matchup_mcts::BattleConditions;
use crate::state::SideMovesFirst;
use crate::state::{Pokemon, PokemonIndex, SideReference, State};
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;

/// Structure to track reasoning steps during matchup analysis
#[derive(Debug, Clone)]
pub struct MatchupReasoning {
    pub s1_name: String,
    pub s2_name: String,
    pub metrics: MatchupMetrics,
    pub win_percentage: f32,
    pub classification: i8,
    pub reasoning_steps: Vec<String>,
    pub primary_reason: String,
}

/// Stores all the computed metrics for the matchup
#[derive(Debug, Clone)]
pub struct MatchupMetrics {
    // Basic Pokemon info
    pub s1_hp: i16,
    pub s1_max_hp: i16,
    pub s2_hp: i16,
    pub s2_max_hp: i16,

    // Moves and damage
    pub s1_best_move: String,
    pub s2_best_move: String,
    pub s1_avg_damage: i16,
    pub s1_max_damage: i16,
    pub s2_avg_damage: i16,
    pub s2_max_damage: i16,

    // KO potential
    pub s1_ohko_chance: f32,
    pub s2_ohko_chance: f32,
    pub s1_turns_to_ko: i32,
    pub s2_turns_to_ko: i32,

    // Priority
    pub s1_priority_damage: i16,
    pub s2_priority_damage: i16,
    pub s1_priority_ko: bool,
    pub s2_priority_ko: bool,

    // Recovery
    pub s1_recovery_per_turn: i16,
    pub s2_recovery_per_turn: i16,
    pub s1_recovery_sufficient: bool,
    pub s1_recovery_dominates: bool,
    pub s2_recovery_sufficient: bool,
    pub s2_recovery_dominates: bool,

    // Setup
    pub s1_has_setup: bool,
    pub s2_has_setup: bool,
    pub s1_can_setup_safely: bool,
    pub s2_can_setup_safely: bool,
    pub s1_boosts_speed: bool,
    pub s2_boosts_speed: bool,
    pub s1_setup_ohko: bool,
    pub s2_setup_ohko: bool,

    // Speed and turn order
    pub s1_moves_first: bool,
    pub speed_tie: bool,
}

impl MatchupReasoning {
    /// Create a new MatchupReasoning instance
    pub fn new(state: &State, s1_idx: PokemonIndex, s2_idx: PokemonIndex) -> Self {
        MatchupReasoning {
            s1_name: state.side_one.pokemon[s1_idx].id.to_string(),
            s2_name: state.side_two.pokemon[s2_idx].id.to_string(),
            metrics: MatchupMetrics::default(),
            win_percentage: 0.0,
            classification: 0,
            reasoning_steps: Vec::new(),
            primary_reason: String::new(),
        }
    }

    /// Add a reasoning step with context
    pub fn add_step(&mut self, step: String) {
        self.reasoning_steps.push(step);
    }

    /// Set the primary reason for the final classification
    pub fn set_primary_reason(&mut self, reason: String) {
        self.primary_reason = reason;
    }

    /// Update the win percentage and classification
    pub fn set_result(&mut self, win_percentage: f32, classification: i8) {
        self.win_percentage = win_percentage;
        self.classification = classification;
    }

    /// Save the detailed reasoning to a file (useful for batch analysis)
    pub fn save_to_file(&self, file_path: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(file_path)?;

        writeln!(
            file,
            "=== MATCHUP: {} vs {} ===",
            self.s1_name, self.s2_name
        )?;
        writeln!(file, "Win Percentage: {:.2}%", self.win_percentage * 100.0)?;
        writeln!(file, "Classification: {}", self.classification_to_string())?;
        writeln!(file, "Primary Reason: {}", self.primary_reason)?;

        writeln!(file, "\n-- KEY METRICS --")?;
        self.metrics.write_to_file(&mut file)?;

        writeln!(file, "\n-- REASONING STEPS --")?;
        for (i, step) in self.reasoning_steps.iter().enumerate() {
            writeln!(file, "{}. {}", i + 1, step)?;
        }

        writeln!(file, "\n\n")?;
        Ok(())
    }

    /// Convert classification value to string description
    fn classification_to_string(&self) -> String {
        match self.classification {
            2 => "COUNTER (Strong Favorable)".to_string(),
            1 => "CHECK (Favorable)".to_string(),
            0 => "NEUTRAL".to_string(),
            -1 => "CHECKED (Unfavorable)".to_string(),
            -2 => "COUNTERED (Strong Unfavorable)".to_string(),
            _ => format!("Unknown ({})", self.classification),
        }
    }

    /// Generate a human-readable summary of the matchup
    pub fn summary(&self) -> String {
        let result = format!(
            "{} vs {} - {} (Win rate: {:.1}%)\n  Primary reason: {}\n  Key metrics: S1 damage: {}, S2 damage: {}, S1 TTK: {}, S2 TTK: {}, {} moves first",
            self.s1_name,
            self.s2_name,
            self.classification_to_string(),
            self.win_percentage * 100.0,
            self.primary_reason,
            self.metrics.s1_avg_damage,
            self.metrics.s2_avg_damage,
            self.metrics.s1_turns_to_ko,
            self.metrics.s2_turns_to_ko,
            if self.metrics.s1_moves_first { self.s1_name.clone() } else { self.s2_name.clone() }
        );
        result
    }
}

impl Default for MatchupMetrics {
    fn default() -> Self {
        MatchupMetrics {
            s1_hp: 0,
            s1_max_hp: 0,
            s2_hp: 0,
            s2_max_hp: 0,
            s1_best_move: String::new(),
            s2_best_move: String::new(),
            s1_avg_damage: 0,
            s1_max_damage: 0,
            s2_avg_damage: 0,
            s2_max_damage: 0,
            s1_ohko_chance: 0.0,
            s2_ohko_chance: 0.0,
            s1_turns_to_ko: 0,
            s2_turns_to_ko: 0,
            s1_priority_damage: 0,
            s2_priority_damage: 0,
            s1_priority_ko: false,
            s2_priority_ko: false,
            s1_recovery_per_turn: 0,
            s2_recovery_per_turn: 0,
            s1_recovery_sufficient: false,
            s1_recovery_dominates: false,
            s2_recovery_sufficient: false,
            s2_recovery_dominates: false,
            s1_has_setup: false,
            s2_has_setup: false,
            s1_can_setup_safely: false,
            s2_can_setup_safely: false,
            s1_boosts_speed: false,
            s2_boosts_speed: false,
            s1_setup_ohko: false,
            s2_setup_ohko: false,
            s1_moves_first: false,
            speed_tie: false,
        }
    }
}

impl MatchupMetrics {
    /// Write metrics to file in a readable format
    fn write_to_file(&self, file: &mut File) -> std::io::Result<()> {
        writeln!(
            file,
            "HP: {} ({}/{}) vs {} ({}/{})",
            self.s1_hp, self.s1_hp, self.s1_max_hp, self.s2_hp, self.s2_hp, self.s2_max_hp
        )?;

        writeln!(
            file,
            "Best Moves: {} vs {}",
            self.s1_best_move, self.s2_best_move
        )?;

        writeln!(
            file,
            "Damage: Avg={}/Max={} vs Avg={}/Max={}",
            self.s1_avg_damage, self.s1_max_damage, self.s2_avg_damage, self.s2_max_damage
        )?;

        writeln!(
            file,
            "OHKO Chance: {:.1}% vs {:.1}%",
            self.s1_ohko_chance * 100.0,
            self.s2_ohko_chance * 100.0
        )?;

        writeln!(
            file,
            "Turns to KO: {} vs {}",
            self.s1_turns_to_ko, self.s2_turns_to_ko
        )?;

        writeln!(
            file,
            "Priority: Damage={} (KO={}) vs Damage={} (KO={})",
            self.s1_priority_damage,
            self.s1_priority_ko,
            self.s2_priority_damage,
            self.s2_priority_ko
        )?;

        writeln!(file, "Recovery: {}/turn (Sufficient={}, Dominates={}) vs {}/turn (Sufficient={}, Dominates={})",
            self.s1_recovery_per_turn, self.s1_recovery_sufficient, self.s1_recovery_dominates,
            self.s2_recovery_per_turn, self.s2_recovery_sufficient, self.s2_recovery_dominates)?;

        writeln!(
            file,
            "Setup: Has={} (Safe={}, Speed={}, OHKO={}) vs Has={} (Safe={}, Speed={}, OHKO={})",
            self.s1_has_setup,
            self.s1_can_setup_safely,
            self.s1_boosts_speed,
            self.s1_setup_ohko,
            self.s2_has_setup,
            self.s2_can_setup_safely,
            self.s2_boosts_speed,
            self.s2_setup_ohko
        )?;

        writeln!(
            file,
            "Speed: S1 Moves First={}, Speed Tie={}",
            self.s1_moves_first, self.speed_tie
        )?;

        Ok(())
    }
}

// Helper function to convert a choice to a string
pub fn choice_to_string(choice: &Choice) -> String {
    format!("{:?}", choice.move_id).to_lowercase()
}

// Modified compute_matchup function that returns detailed reasoning
pub fn compute_matchup_with_reasoning(
    state: &State,
    s1_idx: PokemonIndex,
    s2_idx: PokemonIndex,
    conditions: &BattleConditions,
) -> (i8, MatchupReasoning) {
    // Create a simulation state with the specified conditions
    let mut sim_state = create_simulation_state(state, s1_idx, s2_idx, conditions);

    // Initialize reasoning tracker
    let mut reasoning = MatchupReasoning::new(state, s1_idx, s2_idx);

    // Analyze matchup with detailed reasoning
    let win_percentage = analyze_win_percentage_with_reasoning(&mut sim_state, &mut reasoning);

    // Classify the result
    let classification = classify_matchup_result(win_percentage);
    reasoning.set_result(win_percentage, classification);

    (classification, reasoning)
}

// Enhanced version of analyze_win_percentage that tracks reasoning
pub fn analyze_win_percentage_with_reasoning(
    sim_state: &mut State,
    reasoning: &mut MatchupReasoning,
) -> f32 {
    // Find the best move choices for each Pokémon
    let (s1_best_choice, s1_best_index) = find_best_move(sim_state, &SideReference::SideOne);
    let (s2_best_choice, s2_best_index) = find_best_move(sim_state, &SideReference::SideTwo);

    // Start populating reasoning metrics
    reasoning.metrics.s1_best_move = choice_to_string(&s1_best_choice);
    reasoning.metrics.s2_best_move = choice_to_string(&s2_best_choice);

    // Get setup moves if present
    let s1_has_setup = has_setup_move(sim_state, &SideReference::SideOne);
    let s2_has_setup = has_setup_move(sim_state, &SideReference::SideTwo);
    reasoning.metrics.s1_has_setup = s1_has_setup;
    reasoning.metrics.s2_has_setup = s2_has_setup;

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
        SideMovesFirst::SideOne => {
            reasoning.add_step("Side One moves first due to higher speed".to_string());
            reasoning.metrics.s1_moves_first = true;
            reasoning.metrics.speed_tie = false;
            true
        }
        SideMovesFirst::SideTwo => {
            reasoning.add_step("Side Two moves first due to higher speed".to_string());
            reasoning.metrics.s1_moves_first = false;
            reasoning.metrics.speed_tie = false;
            false
        }
        SideMovesFirst::SpeedTie => {
            reasoning.add_step("Speed tie detected - favoring Side One for analysis".to_string());
            reasoning.metrics.s1_moves_first = true;
            reasoning.metrics.speed_tie = true;
            true
        }
    };

    // Get Pokémon stats
    let s1_pokemon = sim_state.side_one.get_active_immutable();
    let s2_pokemon = sim_state.side_two.get_active_immutable();

    reasoning.metrics.s1_hp = s1_pokemon.hp;
    reasoning.metrics.s1_max_hp = s1_pokemon.maxhp;
    reasoning.metrics.s2_hp = s2_pokemon.hp;
    reasoning.metrics.s2_max_hp = s2_pokemon.maxhp;

    // Calculate damage ranges
    let s1_damage_range = get_damage_range(sim_state, &s1_best_choice, &SideReference::SideOne);
    let s2_damage_range = get_damage_range(sim_state, &s2_best_choice, &SideReference::SideTwo);

    // Calculate recovery amounts (per turn)
    let s1_recovery_per_turn = calculate_recovery_per_turn(sim_state, &SideReference::SideOne);
    let s2_recovery_per_turn = calculate_recovery_per_turn(sim_state, &SideReference::SideTwo);

    reasoning.metrics.s1_recovery_per_turn = s1_recovery_per_turn;
    reasoning.metrics.s2_recovery_per_turn = s2_recovery_per_turn;

    // Calculate priority move damage
    let s1_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideOne);
    let s2_priority_damage = get_best_priority_move_damage(sim_state, &SideReference::SideTwo);

    reasoning.metrics.s1_priority_damage = s1_priority_damage;
    reasoning.metrics.s2_priority_damage = s2_priority_damage;

    // Get average and max damage
    let (s1_avg_damage, s1_max_damage) = if s1_damage_range.is_empty() {
        reasoning.add_step("Side One has no effective damage output".to_string());
        (0, 0)
    } else {
        let avg =
            (s1_damage_range.iter().sum::<i16>() as f32 / s1_damage_range.len() as f32) as i16;
        let max = *s1_damage_range.iter().max().unwrap_or(&0);
        reasoning.add_step(format!(
            "Side One damage output: Avg={}, Max={} ({:.1}% of opponent's HP)",
            avg,
            max,
            (avg as f32 / s2_pokemon.hp as f32) * 100.0
        ));
        (avg, max)
    };

    let (s2_avg_damage, s2_max_damage) = if s2_damage_range.is_empty() {
        reasoning.add_step("Side Two has no effective damage output".to_string());
        (0, 0)
    } else {
        let avg =
            (s2_damage_range.iter().sum::<i16>() as f32 / s2_damage_range.len() as f32) as i16;
        let max = *s2_damage_range.iter().max().unwrap_or(&0);
        reasoning.add_step(format!(
            "Side Two damage output: Avg={}, Max={} ({:.1}% of opponent's HP)",
            avg,
            max,
            (avg as f32 / s1_pokemon.hp as f32) * 100.0
        ));
        (avg, max)
    };

    reasoning.metrics.s1_avg_damage = s1_avg_damage;
    reasoning.metrics.s1_max_damage = s1_max_damage;
    reasoning.metrics.s2_avg_damage = s2_avg_damage;
    reasoning.metrics.s2_max_damage = s2_max_damage;

    // Calculate turns to KO (accounting for recovery)
    let s1_turns_to_ko = calculate_turns_to_ko(s2_pokemon.hp, s1_avg_damage, s2_recovery_per_turn);
    let s2_turns_to_ko = calculate_turns_to_ko(s1_pokemon.hp, s2_avg_damage, s1_recovery_per_turn);

    reasoning.metrics.s1_turns_to_ko = s1_turns_to_ko;
    reasoning.metrics.s2_turns_to_ko = s2_turns_to_ko;

    reasoning.add_step(format!(
        "Turns to KO: Side One={} vs Side Two={}",
        s1_turns_to_ko, s2_turns_to_ko
    ));

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

    reasoning.metrics.s1_ohko_chance = s1_ohko_chance;
    reasoning.metrics.s2_ohko_chance = s2_ohko_chance;

    if s1_ohko_chance > 0.0 {
        reasoning.add_step(format!(
            "Side One OHKO chance: {:.1}%",
            s1_ohko_chance * 100.0
        ));
    }
    if s2_ohko_chance > 0.0 {
        reasoning.add_step(format!(
            "Side Two OHKO chance: {:.1}%",
            s2_ohko_chance * 100.0
        ));
    }

    // Check setup viability
    let s1_can_setup_safely = if s1_has_setup {
        let setup_turns_needed = 1; // Most setup moves need just 1 turn
        let damage_taken_during_setup = setup_turns_needed * s2_max_damage;
        let can_setup = s1_pokemon.hp > damage_taken_during_setup;
        if can_setup {
            reasoning.add_step("Side One can safely set up".to_string());
        } else {
            reasoning.add_step("Side One cannot safely set up".to_string());
        }
        can_setup
    } else {
        false
    };

    let s2_can_setup_safely = if s2_has_setup {
        let setup_turns_needed = 1; // Most setup moves need just 1 turn
        let damage_taken_during_setup = setup_turns_needed * s1_max_damage;
        let can_setup = s2_pokemon.hp > damage_taken_during_setup;
        if can_setup {
            reasoning.add_step("Side Two can safely set up".to_string());
        } else {
            reasoning.add_step("Side Two cannot safely set up".to_string());
        }
        can_setup
    } else {
        false
    };

    reasoning.metrics.s1_can_setup_safely = s1_can_setup_safely;
    reasoning.metrics.s2_can_setup_safely = s2_can_setup_safely;

    // Check post-setup sweep potential
    let s1_boosts_speed = has_speed_boosting_move(sim_state, &SideReference::SideOne);
    let s2_boosts_speed = has_speed_boosting_move(sim_state, &SideReference::SideTwo);

    reasoning.metrics.s1_boosts_speed = s1_boosts_speed;
    reasoning.metrics.s2_boosts_speed = s2_boosts_speed;

    let s1_boosted_damage = s1_avg_damage * 2; // Approximate 1 stage of Attack/SpAtk boost
    let s2_boosted_damage = s2_avg_damage * 2; // Approximate 1 stage of Attack/SpAtk boost

    let s1_setup_ohko = s1_boosted_damage >= s2_pokemon.hp;
    let s2_setup_ohko = s2_boosted_damage >= s1_pokemon.hp;

    reasoning.metrics.s1_setup_ohko = s1_setup_ohko;
    reasoning.metrics.s2_setup_ohko = s2_setup_ohko;

    if s1_has_setup && s1_setup_ohko {
        reasoning.add_step("Side One can OHKO after setup".to_string());
    }
    if s2_has_setup && s2_setup_ohko {
        reasoning.add_step("Side Two can OHKO after setup".to_string());
    }

    // Recovery dominance check (immune to KO due to recovery)
    let s1_recovery_dominates = s1_recovery_per_turn > s2_avg_damage;
    let s2_recovery_dominates = s2_recovery_per_turn > s1_avg_damage;

    reasoning.metrics.s1_recovery_dominates = s1_recovery_dominates;
    reasoning.metrics.s2_recovery_dominates = s2_recovery_dominates;

    if s1_recovery_dominates {
        reasoning.add_step("Side One recovery exceeds Side Two's damage output".to_string());
    }
    if s2_recovery_dominates {
        reasoning.add_step("Side Two recovery exceeds Side One's damage output".to_string());
    }

    // Recovery sufficiency check (can partially offset damage)
    let s1_recovery_sufficient = s1_recovery_per_turn > (s2_avg_damage as f32 * 0.6) as i16;
    let s2_recovery_sufficient = s2_recovery_per_turn > (s1_avg_damage as f32 * 0.6) as i16;

    reasoning.metrics.s1_recovery_sufficient = s1_recovery_sufficient;
    reasoning.metrics.s2_recovery_sufficient = s2_recovery_sufficient;

    if s1_recovery_sufficient {
        reasoning
            .add_step("Side One recovery is sufficient to offset significant damage".to_string());
    }
    if s2_recovery_sufficient {
        reasoning
            .add_step("Side Two recovery is sufficient to offset significant damage".to_string());
    }

    // Check for priority KO
    let s1_priority_ko = s1_priority_damage >= s2_pokemon.hp;
    let s2_priority_ko = s2_priority_damage >= s1_pokemon.hp;

    reasoning.metrics.s1_priority_ko = s1_priority_ko;
    reasoning.metrics.s2_priority_ko = s2_priority_ko;

    if s1_priority_ko {
        reasoning.add_step("Side One can KO with priority move".to_string());
    }
    if s2_priority_ko {
        reasoning.add_step("Side Two can KO with priority move".to_string());
    }

    // Now apply our mathematical classification framework
    let mut win_percentage: f32;

    // COUNTER CONDITIONS (90%+ win rate)
    if s1_recovery_dominates {
        // Recovery exceeds all possible damage from opponent
        win_percentage = 0.95;
        reasoning
            .set_primary_reason("Recovery exceeds all possible damage from opponent".to_string());
        return win_percentage;
    }

    if s1_priority_ko {
        // Can KO with priority before opponent moves
        win_percentage = 0.95;
        reasoning.set_primary_reason("Can KO with priority before opponent moves".to_string());
        return win_percentage;
    }

    if moves_first_side_one && s1_ohko_chance > 0.9 {
        // Has nearly guaranteed OHKO and moves first
        win_percentage = 0.95;
        reasoning.set_primary_reason("Has nearly guaranteed OHKO and moves first".to_string());
        return win_percentage;
    }

    if moves_first_side_one && s1_turns_to_ko <= 2 && s2_turns_to_ko >= 3 {
        // Much faster KO and moves first
        win_percentage = 0.95;
        reasoning.set_primary_reason("Much faster KO and moves first".to_string());
        return win_percentage;
    }

    if s1_can_setup_safely && (s1_boosts_speed || moves_first_side_one) && s1_setup_ohko {
        // Can safely setup and then OHKO
        win_percentage = 0.9;
        reasoning.set_primary_reason("Can safely set up and then OHKO".to_string());
        return win_percentage;
    }

    // CHECK CONDITIONS (70-90% win rate)
    if s1_recovery_sufficient && !s2_recovery_sufficient {
        // Recovery offsets most but not all damage
        win_percentage = 0.8;
        reasoning.set_primary_reason("Recovery offsets most but not all damage".to_string());
        return win_percentage;
    }

    if moves_first_side_one && s1_turns_to_ko <= s2_turns_to_ko {
        // Faster and equal/better KO speed
        win_percentage = 0.75;
        reasoning.set_primary_reason("Faster and equal/better KO speed".to_string());
        return win_percentage;
    }

    if s1_can_setup_safely && s1_setup_ohko && !moves_first_side_one {
        // Can setup but sweeping isn't guaranteed
        win_percentage = 0.7;
        reasoning.set_primary_reason("Can set up and OHKO but doesn't control speed".to_string());
        return win_percentage;
    }

    // NEUTRAL CONDITIONS (30-70% win rate)
    if s1_turns_to_ko == s2_turns_to_ko {
        // Equal KO timing - speed determines winner
        win_percentage = if moves_first_side_one { 0.6 } else { 0.4 };
        reasoning.set_primary_reason("Equal KO timing - speed determines winner".to_string());
        return win_percentage;
    }

    if (moves_first_side_one && s1_turns_to_ko > s2_turns_to_ko)
        || (!moves_first_side_one && s1_turns_to_ko < s2_turns_to_ko)
    {
        // Offsetting advantages
        win_percentage = 0.5;
        reasoning.set_primary_reason(
            "Offsetting advantages (one has speed, other has damage)".to_string(),
        );
        return win_percentage;
    }

    // COUNTERED CONDITIONS (<10% win rate)
    if s2_recovery_dominates {
        // Opponent can recover more than max damage output
        win_percentage = 0.05;
        reasoning
            .set_primary_reason("Opponent can recover more than max damage output".to_string());
        return win_percentage;
    }

    if s2_priority_ko {
        // Opponent KOs with priority before can move
        win_percentage = 0.05;
        reasoning.set_primary_reason("Opponent KOs with priority before can move".to_string());
        return win_percentage;
    }

    if !moves_first_side_one && s2_ohko_chance > 0.9 {
        // Opponent almost guaranteed to OHKO and moves first
        win_percentage = 0.05;
        reasoning
            .set_primary_reason("Opponent almost guaranteed to OHKO and moves first".to_string());
        return win_percentage;
    }

    if !moves_first_side_one && s2_turns_to_ko <= 2 && s1_turns_to_ko >= 3 {
        // Opponent has much faster KO and moves first
        win_percentage = 0.05;
        reasoning.set_primary_reason("Opponent has much faster KO and moves first".to_string());
        return win_percentage;
    }

    if s2_can_setup_safely && (s2_boosts_speed || !moves_first_side_one) && s2_setup_ohko {
        // Opponent can safely set up and OHKO
        win_percentage = 0.1;
        reasoning.set_primary_reason("Opponent can safely set up and OHKO".to_string());
        return win_percentage;
    }

    // CHECKED CONDITIONS (10-30% win rate)
    if !moves_first_side_one && s1_turns_to_ko == s2_turns_to_ko {
        // Equal KO timing but moves second
        win_percentage = 0.25;
        reasoning.set_primary_reason("Equal KO timing but moves second".to_string());
        return win_percentage;
    }

    if s1_turns_to_ko > s2_turns_to_ko && !s1_recovery_sufficient {
        // Takes longer to KO and recovery isn't sufficient
        win_percentage = 0.2;
        reasoning
            .set_primary_reason("Takes longer to KO and recovery isn't sufficient".to_string());
        return win_percentage;
    }
    return 0.5;
}
