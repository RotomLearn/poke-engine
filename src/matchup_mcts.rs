use crate::abilities::Abilities;
use crate::choices::Choices;
use crate::evaluate::evaluate;
use crate::matchup_calc::compute_matchup_fast;
use crate::mcts::{sigmoid, MctsResult, MctsSideResult, Node};
use crate::state::{
    MoveChoice, Pokemon, PokemonIndex, PokemonStatus, Side, State, Terrain, Weather,
};
use std::collections::HashMap;
use std::time::Duration;

// =========== TUNABLE HYPERPARAMETERS ===========

// Evaluation Constants
const COUNTER_BASE_VALUE: f32 = 20.0; // Base value for countering opponents
const COUNTER_COVERAGE_VALUE: f32 = 35.0; // Value for countering percentage of opponent's team
const CHECK_BASE_VALUE: f32 = 10.0; // Base value for checking opponents
const CHECK_COVERAGE_VALUE: f32 = 15.0; // Value for checking percentage of opponent's team
const UNIQUE_COUNTER_BASE_VALUE: f32 = 50.0; // Value for being a unique counter
const UNIQUE_COUNTER_SCALING: f32 = 0.7; // Scaling factor for additional unique counters

// const COUNTER_BASE_VALUE: f32 = 0.0; // Base value for countering opponents
// const COUNTER_COVERAGE_VALUE: f32 = 0.0; // Value for countering percentage of opponent's team
// const CHECK_BASE_VALUE: f32 = 0.0; // Base value for checking opponents
// const CHECK_COVERAGE_VALUE: f32 = 0.0; // Value for checking percentage of opponent's team
// const UNIQUE_COUNTER_BASE_VALUE: f32 = 0.0; // Value for being a unique counter
// const UNIQUE_COUNTER_SCALING: f32 = 0.7; // Scaling factor for additional unique counters

// Cache Evaluation Constants
const CACHE_COVERAGE_THRESHOLD: f32 = 0.7; // Minimum cache coverage required for partial evaluation

// HP Bracket Percentages
const HP_BRACKET_FULL: f32 = 0.9; // 76-100% → use 90%
const HP_BRACKET_HIGH: f32 = 0.65; // 51-75% → use 65%
const HP_BRACKET_MEDIUM: f32 = 0.4; // 26-50% → use 40%
const HP_BRACKET_LOW: f32 = 0.15; // 0-25% → use 15%
                                  // ==============================================

pub struct MatchupCache {
    // Key: (side1_pokemon_index, side2_pokemon_index, battle_conditions_hash)
    // Value: Matchup result (-2 to 2 scale)
    cache: HashMap<(usize, usize, u64), i8>,
}

impl MatchupCache {
    pub fn new() -> Self {
        MatchupCache {
            cache: HashMap::new(),
        }
    }

    fn insert(&mut self, s1_idx: usize, s2_idx: usize, conditions: &BattleConditions, result: i8) {
        let hash = conditions.hash();
        self.cache.insert((s1_idx, s2_idx, hash), result);
    }
}

pub struct BattleConditions {
    pub hp_brackets: [usize; 2], // HP percentage brackets (e.g., [80, 50] for 80-100% and 50-79%)
    pub status: [PokemonStatus; 2], // Status conditions
    pub boosts: [(i8, i8, i8, i8, i8); 2], // (Attack, Defense, Speed) boosts
    pub weather: Weather,
    pub terrain: Terrain,
    pub trick_room: bool,
    pub terastallized: [bool; 2],
}

impl BattleConditions {
    fn from_state(state: &State, s1_idx: PokemonIndex, s2_idx: PokemonIndex) -> Self {
        let s1 = &state.side_one.pokemon[s1_idx];
        let s2 = &state.side_two.pokemon[s2_idx];

        // Only set boosts if index is active
        let s1_boosts = if s1_idx == state.side_one.active_index {
            (
                state.side_one.attack_boost,
                state.side_one.defense_boost,
                state.side_one.special_attack_boost,
                state.side_one.special_defense_boost,
                state.side_one.speed_boost,
            )
        } else {
            (0, 0, 0, 0, 0)
        };

        let s2_boosts = if s2_idx == state.side_two.active_index {
            (
                state.side_two.attack_boost,
                state.side_two.defense_boost,
                state.side_two.special_attack_boost,
                state.side_two.special_defense_boost,
                state.side_two.speed_boost,
            )
        } else {
            (0, 0, 0, 0, 0)
        };

        BattleConditions {
            hp_brackets: [
                get_hp_bracket(s1.hp as f32 / s1.maxhp as f32),
                get_hp_bracket(s2.hp as f32 / s2.maxhp as f32),
            ],
            status: [s1.status, s2.status],
            boosts: [s1_boosts, s2_boosts],
            weather: state.weather.weather_type,
            terrain: state.terrain.terrain_type,
            trick_room: state.trick_room.active,
            terastallized: [s1.terastallized, s2.terastallized],
        }
    }

    fn hash(&self) -> u64 {
        let mut hash = 0u64;

        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.hp_brackets[0] as u64);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.hp_brackets[1] as u64);
        hash = hash.wrapping_mul(31).wrapping_add(self.status[0] as u64);
        hash = hash.wrapping_mul(31).wrapping_add(self.status[1] as u64);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[0].0 as u64 + 6);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[0].1 as u64 + 6);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[0].2 as u64 + 6);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[1].0 as u64 + 6);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[1].1 as u64 + 6);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.boosts[1].2 as u64 + 6);
        hash = hash.wrapping_mul(31).wrapping_add(self.weather as u64);
        hash = hash.wrapping_mul(31).wrapping_add(self.terrain as u64);
        hash = hash.wrapping_mul(31).wrapping_add(self.trick_room as u64);

        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.terastallized[0] as u64);
        hash = hash
            .wrapping_mul(31)
            .wrapping_add(self.terastallized[1] as u64);

        hash
    }
}

fn get_hp_bracket(hp_percentage: f32) -> usize {
    if hp_percentage > 0.75 {
        0
    }
    // 76-100%
    else if hp_percentage > 0.5 {
        1
    }
    // 51-75%
    else if hp_percentage > 0.25 {
        2
    }
    // 26-50%
    else {
        3
    } // 0-25%
}

pub fn create_simulation_state(
    state: &State,
    s1_idx: PokemonIndex,
    s2_idx: PokemonIndex,
    conditions: &BattleConditions,
) -> State {
    let mut sim_state = State::default();

    // Copy the selected Pokémon
    sim_state.side_one.pokemon[PokemonIndex::P0] = state.side_one.pokemon[s1_idx].clone();
    sim_state.side_two.pokemon[PokemonIndex::P0] = state.side_two.pokemon[s2_idx].clone();

    // Set them as active
    sim_state.side_one.active_index = PokemonIndex::P0;
    sim_state.side_two.active_index = PokemonIndex::P0;

    if !state.side_one.can_use_tera() {
        sim_state.side_one.pokemon[PokemonIndex::P5].terastallized = true;
    }
    if !state.side_two.can_use_tera() {
        sim_state.side_two.pokemon[PokemonIndex::P5].terastallized = true;
    }

    // Apply the battle conditions
    apply_conditions(&mut sim_state, conditions);
    sim_state
}

pub fn apply_conditions(state: &mut State, conditions: &BattleConditions) {
    // Apply HP percentages
    let s1_active = state.side_one.get_active();
    let s2_active = state.side_two.get_active();

    // HP conditions
    adjust_hp_for_bracket(s1_active, conditions.hp_brackets[0]);
    adjust_hp_for_bracket(s2_active, conditions.hp_brackets[1]);

    // Status conditions
    s1_active.status = conditions.status[0];
    s2_active.status = conditions.status[1];

    s1_active.terastallized = conditions.terastallized[0];
    s2_active.terastallized = conditions.terastallized[1];

    // Stat boosts
    state.side_one.attack_boost = conditions.boosts[0].0;
    state.side_one.defense_boost = conditions.boosts[0].1;
    state.side_one.speed_boost = conditions.boosts[0].2;

    state.side_two.attack_boost = conditions.boosts[1].0;
    state.side_two.defense_boost = conditions.boosts[1].1;
    state.side_two.speed_boost = conditions.boosts[1].2;

    // Field conditions
    state.weather.weather_type = conditions.weather;
    state.weather.turns_remaining = 5;

    state.terrain.terrain_type = conditions.terrain;
    state.terrain.turns_remaining = 5;

    state.trick_room.active = conditions.trick_room;
    state.trick_room.turns_remaining = if conditions.trick_room { 5 } else { 0 };
}

pub fn adjust_hp_for_bracket(pokemon: &mut Pokemon, bracket: usize) {
    let percentage = match bracket {
        0 => HP_BRACKET_FULL,   // 76-100% → use 90%
        1 => HP_BRACKET_HIGH,   // 51-75% → use 65%
        2 => HP_BRACKET_MEDIUM, // 26-50% → use 40%
        3 => HP_BRACKET_LOW,    // 0-25% → use 15%
        _ => 1.0,
    };

    pokemon.hp = (pokemon.maxhp as f32 * percentage) as i16;
}

pub fn initialize_matchup_cache(state: &State) -> MatchupCache {
    let mut cache = MatchupCache::new();

    // Get all alive Pokémon
    let s1_alive = get_alive_indices(&state.side_one);
    let s2_alive = get_alive_indices(&state.side_two);
    let possible_conditions = determine_possible_conditions_for_matchup(state);
    // For each pair of Pokémon, determine possible conditions specific to that matchup
    for &s1_idx in &s1_alive {
        for &s2_idx in &s2_alive {
            for conditions in &possible_conditions {
                let result = compute_matchup_fast(state, s1_idx, s2_idx, conditions);
                cache.insert(s1_idx as usize, s2_idx as usize, conditions, result);
            }
        }
    }

    cache
}

pub fn analyze_matchup_cache(state: &State, cache: &mut TeamMatchupCache) {
    // Get all alive Pokémon
    let s1_alive = get_alive_indices(&state.side_one);
    let s2_alive = get_alive_indices(&state.side_two);

    println!("Matchup Cache Analysis");
    println!("=====================\n");

    // 1. Basic statistics
    let total_matchups = cache.cache.len();
    println!("Total cached matchups: {}", total_matchups);

    // 2. Matchup matrix - baseline conditions
    println!("\nBaseline Matchup Matrix (Full HP, No Status, No Boosts):");
    println!("----------------------------------------------------------");

    // Header row with side 2 Pokémon names
    print!("           |");
    for &s2_idx in &s2_alive {
        let s2_name = &state.side_two.pokemon[s2_idx].id.to_string();
        print!(" {:<10} |", truncate_name(s2_name, 10));
    }
    println!();

    // Print divider
    print!("-----------|");
    for _ in &s2_alive {
        print!("------------|");
    }
    println!();

    // Print matchup data for each side 1 Pokémon
    for &s1_idx in &s1_alive {
        let s1_name = &state.side_one.pokemon[s1_idx].id.to_string();
        print!("{:<11}|", truncate_name(s1_name, 11));

        for &s2_idx in &s2_alive {
            // Get baseline matchup
            let baseline_conditions = BattleConditions {
                hp_brackets: [0, 0], // Full HP
                status: [PokemonStatus::NONE, PokemonStatus::NONE],
                boosts: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)],
                weather: Weather::NONE,
                terrain: Terrain::NONE,
                trick_room: false,
                terastallized: [false, false],
            };

            let result = cache
                .get(s1_idx as usize, s2_idx as usize, &baseline_conditions)
                .unwrap_or(0);

            let result_str = match result {
                2 => " COUNTER   ",
                1 => " Check     ",
                0 => " Neutral   ",
                -1 => " Checked   ",
                -2 => " COUNTERED ",
                _ => " Unknown   ",
            };

            print!("|{}", result_str);
        }
        println!("|");
    }

    // 3. Find counters and vulnerabilities
    println!("\nTeam Counters and Vulnerabilities:");
    println!("----------------------------------");

    // For each side 1 Pokémon, which side 2 Pokémon they counter/are countered by
    for &s1_idx in &s1_alive {
        let s1_name = &state.side_one.pokemon[s1_idx].id.to_string();

        // Find Pokémon that s1_idx counters
        let counters: Vec<String> = s2_alive
            .iter()
            .filter(|&&s2_idx| {
                let baseline_conditions = BattleConditions {
                    hp_brackets: [0, 0],
                    status: [PokemonStatus::NONE, PokemonStatus::NONE],
                    boosts: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)],
                    weather: Weather::NONE,
                    terrain: Terrain::NONE,
                    trick_room: false,
                    terastallized: [false, false],
                };

                cache
                    .get(s1_idx as usize, s2_idx as usize, &baseline_conditions)
                    .unwrap_or(0)
                    >= 2
            })
            .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
            .collect();

        // Find Pokémon that counter s1_idx
        let countered_by: Vec<String> = s2_alive
            .iter()
            .filter(|&&s2_idx| {
                let baseline_conditions = BattleConditions {
                    hp_brackets: [0, 0],
                    status: [PokemonStatus::NONE, PokemonStatus::NONE],
                    boosts: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)],
                    weather: Weather::NONE,
                    terrain: Terrain::NONE,
                    trick_room: false,
                    terastallized: [false, false],
                };

                cache
                    .get(s1_idx as usize, s2_idx as usize, &baseline_conditions)
                    .unwrap_or(0)
                    <= -2
            })
            .map(|&s2_idx| state.side_two.pokemon[s2_idx].id.to_string())
            .collect();

        println!("{}: ", s1_name);
        let counters_str = if counters.is_empty() {
            "None".to_string()
        } else {
            counters.join(", ")
        };
        let countered_by_str = if countered_by.is_empty() {
            "None".to_string()
        } else {
            countered_by.join(", ")
        };
        println!("  Counters: {}", counters_str);
        println!("  Countered by: {}", countered_by_str);
    }
}

// Helper functions
fn truncate_name(name: &str, max_length: usize) -> String {
    if name.len() <= max_length {
        name.to_string()
    } else {
        format!("{}...", &name[0..(max_length - 3)])
    }
}

pub fn determine_possible_conditions_for_matchup(state: &State) -> Vec<BattleConditions> {
    // Get possible parameter values specific to this matchup
    let possible_hp_brackets = vec![0, 1, 2, 3]; // 76-100%, 51-75%, 26-50%, 0-25%
    let possible_weather = get_possible_weather(state);
    let possible_terrain = get_possible_terrain(state);

    let trick_room_options = if has_trick_room(state) {
        vec![false, true]
    } else {
        vec![false]
    };

    let s1_tera_options = if state.side_one.can_use_tera() {
        vec![false, true]
    } else {
        vec![false]
    };

    let s2_tera_options = if state.side_two.can_use_tera() {
        vec![false, true]
    } else {
        vec![false]
    };

    let mut conditions = Vec::new();

    for &hp1 in &possible_hp_brackets {
        for &hp2 in &possible_hp_brackets {
            for &weather in &possible_weather {
                for &terrain in &possible_terrain {
                    for &trick_room in &trick_room_options {
                        for &s1_tera in &s1_tera_options {
                            for &s2_tera in &s2_tera_options {
                                conditions.push(BattleConditions {
                                    hp_brackets: [hp1, hp2],
                                    status: [PokemonStatus::NONE, PokemonStatus::NONE],
                                    boosts: [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)],
                                    weather,
                                    terrain,
                                    trick_room,
                                    terastallized: [s1_tera, s2_tera],
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    conditions
}

fn get_possible_weather(state: &State) -> Vec<Weather> {
    let mut weather_types = vec![Weather::NONE];

    // Check for weather-setting abilities
    for side in [&state.side_one, &state.side_two] {
        for pokemon in side.pokemon.into_iter() {
            match pokemon.ability {
                Abilities::DROUGHT | Abilities::ORICHALCUMPULSE => {
                    weather_types.push(Weather::SUN);
                }
                Abilities::DRIZZLE => {
                    weather_types.push(Weather::RAIN);
                }
                Abilities::SANDSTREAM => {
                    weather_types.push(Weather::SAND);
                }
                Abilities::SNOWWARNING => {
                    #[cfg(feature = "gen9")]
                    weather_types.push(Weather::SNOW);
                    #[cfg(not(feature = "gen9"))]
                    weather_types.push(Weather::HAIL);
                }
                Abilities::DESOLATELAND => {
                    weather_types.push(Weather::HARSHSUN);
                }
                Abilities::PRIMORDIALSEA => {
                    weather_types.push(Weather::HEAVYRAIN);
                }
                _ => {}
            }
        }
    }

    // Check for weather-setting moves
    for side in [&state.side_one, &state.side_two] {
        for pokemon in side.pokemon.into_iter() {
            for mv in pokemon.moves.into_iter() {
                match mv.id {
                    Choices::SUNNYDAY => {
                        weather_types.push(Weather::SUN);
                    }
                    Choices::RAINDANCE => {
                        weather_types.push(Weather::RAIN);
                    }
                    Choices::SANDSTORM => {
                        weather_types.push(Weather::SAND);
                    }
                    Choices::HAIL => {
                        weather_types.push(Weather::HAIL);
                    }
                    Choices::SNOWSCAPE | Choices::CHILLYRECEPTION => {
                        weather_types.push(Weather::SNOW);
                    }
                    _ => {}
                }
            }
        }
    }

    // Remove duplicates
    weather_types.sort();
    weather_types.dedup();

    weather_types
}

fn get_possible_terrain(state: &State) -> Vec<Terrain> {
    let mut terrain_types = vec![Terrain::NONE];

    // Check for terrain-setting abilities
    for side in [&state.side_one, &state.side_two] {
        for pokemon in side.pokemon.into_iter() {
            match pokemon.ability {
                Abilities::ELECTRICSURGE | Abilities::HADRONENGINE => {
                    terrain_types.push(Terrain::ELECTRICTERRAIN);
                }
                Abilities::PSYCHICSURGE => {
                    terrain_types.push(Terrain::PSYCHICTERRAIN);
                }
                Abilities::MISTYSURGE => {
                    terrain_types.push(Terrain::MISTYTERRAIN);
                }
                Abilities::GRASSYSURGE => {
                    terrain_types.push(Terrain::GRASSYTERRAIN);
                }
                _ => {}
            }
        }
    }

    // Check for terrain-setting moves
    for side in [&state.side_one, &state.side_two] {
        for pokemon in side.pokemon.into_iter() {
            for mv in pokemon.moves.into_iter() {
                match mv.id {
                    Choices::ELECTRICTERRAIN => {
                        terrain_types.push(Terrain::ELECTRICTERRAIN);
                    }
                    Choices::PSYCHICTERRAIN => {
                        terrain_types.push(Terrain::PSYCHICTERRAIN);
                    }
                    Choices::MISTYTERRAIN => {
                        terrain_types.push(Terrain::MISTYTERRAIN);
                    }
                    Choices::GRASSYTERRAIN => {
                        terrain_types.push(Terrain::GRASSYTERRAIN);
                    }
                    _ => {}
                }
            }
        }
    }

    // Remove duplicates
    terrain_types.sort();
    terrain_types.dedup();

    terrain_types
}

fn has_trick_room(state: &State) -> bool {
    let mut trick_room = false;

    // Check all opponent's Pokémon for stat-lowering moves
    for side in [&state.side_one, &state.side_two] {
        for pokemon in side.pokemon.into_iter() {
            if pokemon.hp <= 0 {
                continue; // Skip fainted Pokémon
            }

            for mv in pokemon.moves.into_iter() {
                match mv.id {
                    // Attack-lowering moves
                    Choices::TRICKROOM => {
                        trick_room = true;
                    }
                    _ => {}
                }
            }
        }
    }
    trick_room
}

fn get_alive_indices(side: &Side) -> Vec<PokemonIndex> {
    let mut vec = Vec::with_capacity(6);
    let mut iter = side.pokemon.into_iter();

    while let Some(p) = iter.next() {
        if p.hp > 0 {
            vec.push(iter.pokemon_index.clone());
        }
    }

    vec
}

// Define the matchup relationship between two Pokémon
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchupRelationship {
    StrongCounter = 2,    // Strongly favorable (counter)
    WeakCounter = 1,      // Favorable (check)
    Neutral = 0,          // Neutral matchup
    WeakCountered = -1,   // Unfavorable (checked)
    StrongCountered = -2, // Strongly unfavorable (hard countered)
}

impl From<i8> for MatchupRelationship {
    fn from(value: i8) -> Self {
        match value {
            2 => MatchupRelationship::StrongCounter,
            1 => MatchupRelationship::WeakCounter,
            0 => MatchupRelationship::Neutral,
            -1 => MatchupRelationship::WeakCountered,
            -2 => MatchupRelationship::StrongCountered,
            _ => MatchupRelationship::Neutral,
        }
    }
}

// Structure to hold precomputed matchup results for the current battle state
struct CurrentMatchups {
    // Maps each Pokémon to its counters, checks, and unique counters
    s1_counter_info: HashMap<usize, PokemonCounterInfo>,
    s2_counter_info: HashMap<usize, PokemonCounterInfo>,
}

// Stores all counter-related information for a single Pokémon
pub struct PokemonCounterInfo {
    counters: Vec<usize>,        // Pokémon this one counters
    checks: Vec<usize>,          // Pokémon this one checks
    countered_by: Vec<usize>,    // Pokémon that counter this one
    checked_by: Vec<usize>,      // Pokémon that check this one
    unique_counters: Vec<usize>, // Opponents for which this is the only counter
}

// Define an enum for different types of evaluations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationType {
    Complete, // All matchups found in cache
    Partial,  // Some matchups found, above threshold
    Fallback, // Not enough matchups found, using base score only
}

impl PokemonCounterInfo {
    fn new() -> Self {
        PokemonCounterInfo {
            counters: Vec::new(),
            checks: Vec::new(),
            countered_by: Vec::new(),
            checked_by: Vec::new(),
            unique_counters: Vec::new(),
        }
    }
}

// Optimized TeamMatchupCache
pub struct TeamMatchupCache {
    // Original matchup cache with condition-specific results
    // Key: (s1_idx, s2_idx, conditions_hash), Value: Matchup result (-2 to 2 scale)
    cache: HashMap<(usize, usize, u64), i8>,
    hits: usize,
    misses: usize,
    total_queries: usize,
    team_strategic_value_sum: f32,
    eval_count: usize,
    team_base_value_sum: f32,
    pub team_base_value_mean: f32,
    pub team_strategic_value_mean: f32,
    pub complete_evaluations: usize,
    pub partial_evaluations: usize,
    pub fallbacks: usize,
    pub total_evaluations: usize,
}

impl TeamMatchupCache {
    pub fn new() -> Self {
        TeamMatchupCache {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            total_queries: 0,
            team_strategic_value_sum: 0.0,
            team_strategic_value_mean: 0.0,
            team_base_value_sum: 0.0,
            eval_count: 0,
            team_base_value_mean: 0.0,
            complete_evaluations: 0,
            partial_evaluations: 0,
            fallbacks: 0,
            total_evaluations: 0,
        }
    }

    pub fn update_team_value_mean(&mut self, strat_value: f32, base_value: f32) {
        self.team_strategic_value_sum += strat_value;
        self.team_base_value_sum += base_value;
        self.eval_count += 1;
        self.team_strategic_value_mean = self.team_strategic_value_sum / self.eval_count as f32;
        self.team_base_value_mean = self.team_base_value_sum / self.eval_count as f32;
    }

    // Add this method to update statistics
    pub fn log_evaluation_type(&mut self, eval_type: EvaluationType) {
        self.total_evaluations += 1;
        match eval_type {
            EvaluationType::Complete => self.complete_evaluations += 1,
            EvaluationType::Partial => self.partial_evaluations += 1,
            EvaluationType::Fallback => self.fallbacks += 1,
        }
    }

    // Method to get statistics
    pub fn get_evaluation_stats(&self) -> (usize, usize, usize, usize, f32, f32, f32) {
        let complete_rate = if self.total_evaluations > 0 {
            self.complete_evaluations as f32 / self.total_evaluations as f32 * 100.0
        } else {
            0.0
        };

        let partial_rate = if self.total_evaluations > 0 {
            self.partial_evaluations as f32 / self.total_evaluations as f32 * 100.0
        } else {
            0.0
        };

        let fallback_rate = if self.total_evaluations > 0 {
            self.fallbacks as f32 / self.total_evaluations as f32 * 100.0
        } else {
            0.0
        };

        (
            self.complete_evaluations,
            self.partial_evaluations,
            self.fallbacks,
            self.total_evaluations,
            complete_rate,
            partial_rate,
            fallback_rate,
        )
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    // Basic cache operations
    pub fn get(
        &mut self,
        s1_idx: usize,
        s2_idx: usize,
        conditions: &BattleConditions,
    ) -> Option<i8> {
        self.total_queries += 1;
        let hash = conditions.hash();
        match self.cache.get(&(s1_idx, s2_idx, hash)) {
            Some(value) => {
                self.hits += 1;
                Some(*value)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }
    pub fn get_stats(&self) -> (usize, usize, usize, f32) {
        let hit_rate = if self.total_queries > 0 {
            self.hits as f32 / self.total_queries as f32 * 100.0
        } else {
            0.0
        };
        (self.hits, self.misses, self.total_queries, hit_rate)
    }

    pub fn insert(
        &mut self,
        s1_idx: usize,
        s2_idx: usize,
        conditions: &BattleConditions,
        result: i8,
    ) {
        let hash = conditions.hash();
        self.cache.insert((s1_idx, s2_idx, hash), result);
    }

    // Precompute all matchups for the current battle state
    // Returns the computed matchups and any new cache entries that should be added
    fn precompute_current_matchups(
        &mut self,
        state: &State,
    ) -> (CurrentMatchups, Vec<((usize, usize, u64), i8)>) {
        let s1_alive = get_alive_indices(&state.side_one);
        let s2_alive = get_alive_indices(&state.side_two);

        let mut matchup_results = HashMap::new();
        let mut s1_counter_info = HashMap::new();
        let mut s2_counter_info = HashMap::new();

        // Initialize counter info for each Pokémon
        for &s1_idx in &s1_alive {
            s1_counter_info.insert(s1_idx as usize, PokemonCounterInfo::new());
        }

        for &s2_idx in &s2_alive {
            s2_counter_info.insert(s2_idx as usize, PokemonCounterInfo::new());
        }

        // Collect all the new cache entries that need to be added
        let new_cache_entries = Vec::new();

        // Precompute all matchups using current conditions
        for &s1_idx in &s1_alive {
            let s1_usize = s1_idx as usize;

            for &s2_idx in &s2_alive {
                let s2_usize = s2_idx as usize;

                // Create condition specific to this matchup
                let current_conditions = BattleConditions::from_state(state, s1_idx, s2_idx);

                // Get or compute the matchup result
                let result = match self.get(s1_usize, s2_usize, &current_conditions) {
                    Some(r) => r,
                    None => {
                        // // Compute the result
                        // let r = compute_matchup(state, s1_idx, s2_idx, &current_conditions);
                        // // Store it for later caching
                        // new_cache_entries
                        //     .push(((s1_usize, s2_usize, current_conditions.hash()), r));
                        // r
                        continue;
                    }
                };

                // Store the result
                matchup_results.insert((s1_usize, s2_usize), result);

                // Update counter information based on the result
                match result {
                    2 => {
                        // s1 strongly counters s2
                        s1_counter_info
                            .get_mut(&s1_usize)
                            .unwrap()
                            .counters
                            .push(s2_usize);
                        s2_counter_info
                            .get_mut(&s2_usize)
                            .unwrap()
                            .countered_by
                            .push(s1_usize);
                    }
                    1 => {
                        // s1 checks s2
                        s1_counter_info
                            .get_mut(&s1_usize)
                            .unwrap()
                            .checks
                            .push(s2_usize);
                        s2_counter_info
                            .get_mut(&s2_usize)
                            .unwrap()
                            .checked_by
                            .push(s1_usize);
                    }
                    -1 => {
                        // s2 checks s1
                        s2_counter_info
                            .get_mut(&s2_usize)
                            .unwrap()
                            .checks
                            .push(s1_usize);
                        s1_counter_info
                            .get_mut(&s1_usize)
                            .unwrap()
                            .checked_by
                            .push(s2_usize);
                    }
                    -2 => {
                        // s2 strongly counters s1
                        s2_counter_info
                            .get_mut(&s2_usize)
                            .unwrap()
                            .counters
                            .push(s1_usize);
                        s1_counter_info
                            .get_mut(&s1_usize)
                            .unwrap()
                            .countered_by
                            .push(s2_usize);
                    }
                    _ => {} // Neutral matchup
                }
            }
        }

        // Find unique counters
        // For each opponent Pokémon, check if it has exactly one counter
        for &s2_idx in &s2_alive {
            let s2_usize = s2_idx as usize;
            let countered_by = &s2_counter_info[&s2_usize].countered_by;

            if countered_by.len() == 1 {
                // This opponent has exactly one counter
                let unique_counter = countered_by[0];
                s1_counter_info
                    .get_mut(&unique_counter)
                    .unwrap()
                    .unique_counters
                    .push(s2_usize);
            }
        }

        // Same for side 2's unique counters
        for &s1_idx in &s1_alive {
            let s1_usize = s1_idx as usize;
            let countered_by = &s1_counter_info[&s1_usize].countered_by;

            if countered_by.len() == 1 {
                // This opponent has exactly one counter
                let unique_counter = countered_by[0];
                s2_counter_info
                    .get_mut(&unique_counter)
                    .unwrap()
                    .unique_counters
                    .push(s1_usize);
            }
        }

        (
            CurrentMatchups {
                s1_counter_info,
                s2_counter_info,
            },
            new_cache_entries,
        )
    }

    // Calculate strategic value using the precomputed matchups
    pub fn calculate_team_strategic_value(&mut self, state: &State, base_value: f32) -> f32 {
        let s1_alive = get_alive_indices(&state.side_one);
        let s2_alive = get_alive_indices(&state.side_two);

        // Early return if one team has no Pokémon
        if s1_alive.is_empty() || s2_alive.is_empty() {
            return 0.0;
        }

        // Precompute all matchups for the current state
        let (matchups, new_cache_entries) = self.precompute_current_matchups(state);

        // Add any newly computed results to the cache
        for ((s1_idx, s2_idx, hash), result) in new_cache_entries {
            self.cache.insert((s1_idx, s2_idx, hash), result);
        }

        let mut strategic_value = 0.0;

        // Evaluate side 1's strategic position
        for &s1_idx in &s1_alive {
            let s1_usize = s1_idx as usize;

            if let Some(info) = matchups.s1_counter_info.get(&s1_usize) {
                // Add value for countering opponent Pokémon
                strategic_value += evaluate_counter_value(info.counters.len(), s2_alive.len());

                // Add value for checking opponent Pokémon
                strategic_value += evaluate_check_value(info.checks.len(), s2_alive.len());

                // Add value for being a unique counter
                strategic_value += evaluate_uniqueness_value(info.unique_counters.len());

                // Subtract value for being countered by opponent Pokémon
                strategic_value -= evaluate_counter_value(info.countered_by.len(), s2_alive.len());

                // Subtract value for being checked by opponent Pokémon
                strategic_value -= evaluate_check_value(info.checked_by.len(), s2_alive.len());
            }
        }

        // Evaluate side 2's strategic position (inverse of side 1's perspective)
        for &s2_idx in &s2_alive {
            let s2_usize = s2_idx as usize;

            if let Some(info) = matchups.s2_counter_info.get(&s2_usize) {
                // Add value for countering opponent Pokémon (negative for side 1's perspective)
                strategic_value -= evaluate_counter_value(info.counters.len(), s1_alive.len());

                // Add value for checking opponent Pokémon (negative for side 1's perspective)
                strategic_value -= evaluate_check_value(info.checks.len(), s1_alive.len());

                // Add value for being a unique counter (negative for side 1's perspective)
                strategic_value -= evaluate_uniqueness_value(info.unique_counters.len());

                // Subtract value for being countered by opponent Pokémon (positive for side 1's perspective)
                strategic_value += evaluate_counter_value(info.countered_by.len(), s1_alive.len());

                // Subtract value for being checked by opponent Pokémon (positive for side 1's perspective)
                strategic_value += evaluate_check_value(info.checked_by.len(), s1_alive.len());
            }
        }

        self.update_team_value_mean(strategic_value, base_value);

        strategic_value
    }
}

// Evaluation functions
// Evaluate the strategic value of a Pokémon that strongly counters N opponent Pokémon
fn evaluate_counter_value(num_countered: usize, total_opponents: usize) -> f32 {
    if num_countered == 0 || total_opponents == 0 {
        return 0.0;
    }

    // Base value for countering any number of opponents
    let base_value = COUNTER_BASE_VALUE;

    // Value for coverage - how much of the opponent's team this Pokémon counters
    let coverage_ratio = num_countered as f32 / total_opponents as f32;
    let coverage_value = COUNTER_COVERAGE_VALUE * coverage_ratio;

    base_value + coverage_value
}

// Evaluate the strategic value of a Pokémon that checks (but doesn't counter) N opponent Pokémon
fn evaluate_check_value(num_checked: usize, total_opponents: usize) -> f32 {
    if num_checked == 0 || total_opponents == 0 {
        return 0.0;
    }

    // Base value for checking any number of opponents (lower than counter value)
    let base_value = CHECK_BASE_VALUE;

    // Value for coverage - how much of the opponent's team this Pokémon checks
    let coverage_ratio = num_checked as f32 / total_opponents as f32;
    let coverage_value = CHECK_COVERAGE_VALUE * coverage_ratio;

    base_value + coverage_value
}

// Calculate value for being a unique counter to specific threats
fn evaluate_uniqueness_value(opponents_unique_countered: usize) -> f32 {
    if opponents_unique_countered == 0 {
        return 0.0;
    }

    // High value for being the only counter to any opponent
    let base_unique_value = UNIQUE_COUNTER_BASE_VALUE;

    // Additional value for each additional uniquely countered opponent
    // Use diminishing returns to prevent excessive scaling
    let scaling_factor = 1.0 + (opponents_unique_countered as f32 - 1.0) * UNIQUE_COUNTER_SCALING;

    base_unique_value * scaling_factor
}

// Modified MCTS rollout to use team-based evaluation
fn do_mcts_with_team_matchups(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    matchup_cache: &mut TeamMatchupCache,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };

    // Use the team-based evaluation for the current state
    let battle_is_over = state.battle_is_over();
    let rollout_result = if battle_is_over == 0.0 {
        let current_eval = evaluate_with_team_matchups(state, matchup_cache);
        sigmoid(current_eval - *root_eval)
    } else {
        if battle_is_over == -1.0 {
            0.0
        } else {
            battle_is_over
        }
    };

    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

// MCTS function that uses condition-aware team-based matchup evaluation
pub fn perform_mcts_with_team_matchups(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    matchup_cache: &mut TeamMatchupCache,
) -> MctsResult {
    let mut root_node = Node::new(side_one_options, side_two_options);
    root_node.root = true;

    // Evaluate the root state using our team-based evaluation
    let root_eval = evaluate_with_team_matchups(state, matchup_cache);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < max_time {
        for _ in 0..1000 {
            do_mcts_with_team_matchups(&mut root_node, state, &root_eval, matchup_cache);
        }

        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let max_depth = root_node.get_max_depth();
    let result = MctsResult {
        s1: root_node
            .s1_options
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
        max_depth,
    };

    result
}

fn evaluate_with_team_matchups(state: &State, matchup_cache: &mut TeamMatchupCache) -> f32 {
    // Base evaluation
    let base_score = evaluate(state);

    // Get all alive Pokémon
    let s1_alive = get_alive_indices(&state.side_one);
    let s2_alive = get_alive_indices(&state.side_two);

    // Early return if either side has no Pokémon
    if s1_alive.is_empty() || s2_alive.is_empty() {
        return base_score;
    }

    // First calculate cache coverage
    let mut total_matchups = 0;
    let mut cached_matchups = 0;

    // Count cached matchups for coverage calculation
    for &s1_idx in &s1_alive {
        for &s2_idx in &s2_alive {
            total_matchups += 1;
            let conditions = BattleConditions::from_state(state, s1_idx, s2_idx);

            if matchup_cache
                .get(s1_idx as usize, s2_idx as usize, &conditions)
                .is_some()
            {
                cached_matchups += 1;
            }
        }
    }

    // Calculate coverage
    let coverage = if total_matchups > 0 {
        cached_matchups as f32 / total_matchups as f32
    } else {
        0.0
    };

    // Determine evaluation type based on coverage
    let threshold = CACHE_COVERAGE_THRESHOLD;
    let eval_type = if coverage == 1.0 {
        EvaluationType::Complete
    } else if coverage >= threshold {
        EvaluationType::Partial
    } else {
        EvaluationType::Fallback
    };

    // Log evaluation type
    matchup_cache.log_evaluation_type(eval_type);

    // Now get the strategic value using the more sophisticated method
    let strategic_value = if eval_type == EvaluationType::Fallback {
        0.0 // Skip calculation if we're going to fall back anyway
    } else {
        matchup_cache.calculate_team_strategic_value(state, base_score)
    };

    // Return appropriate score based on evaluation type
    match eval_type {
        EvaluationType::Complete => {
            // Full strategic value
            base_score + strategic_value
        }
        EvaluationType::Partial => {
            base_score
                + (strategic_value * coverage)
                + matchup_cache.team_strategic_value_mean * (1. - coverage)
        }
        EvaluationType::Fallback => {
            // Use base score plus mean team strategic value
            base_score + matchup_cache.team_strategic_value_mean
        }
    }
}

pub fn initialize_team_matchup_cache(state: &State) -> TeamMatchupCache {
    let mut cache = TeamMatchupCache::new();

    // Get all alive Pokémon
    let s1_alive = get_alive_indices(&state.side_one);
    let s2_alive = get_alive_indices(&state.side_two);

    // Get possible battle conditions
    let possible_conditions = determine_possible_conditions_for_matchup(state);

    // For each pair of Pokémon, compute matchups under various conditions
    for &s1_idx in &s1_alive {
        for &s2_idx in &s2_alive {
            for conditions in &possible_conditions {
                // Compute matchup value
                let result = compute_matchup_fast(state, s1_idx, s2_idx, conditions);
                cache.insert(s1_idx as usize, s2_idx as usize, conditions, result);
            }
        }
    }

    cache
}
