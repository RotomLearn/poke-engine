#![allow(non_snake_case)]
use crate::choices;
use crate::choices::{Choice, Choices, MoveCategory, MOVES};
use crate::embedding;
use crate::evaluate::evaluate;
use crate::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move_pair,
};
use crate::inspect_state::generate_observation_output;
use crate::instruction::{Instruction, StateInstructions};
use crate::matchup_mcts::initialize_team_matchup_cache;
use crate::matchup_mcts::{analyze_matchup_cache, perform_mcts_with_team_matchups};
use crate::matchup_visualization_tool::MatchupVisualizer;
use crate::mcts::{perform_mcts, MctsResult};
use crate::mcts_pn::{perform_mcts_pn_batched, MctsResultPN, PolicyNetwork};
use crate::mcts_pruned::{perform_mcts_pruned_batched, JointNetwork, MctsResultPruned};
use crate::search::{expectiminimax_search, iterative_deepen_expectiminimax, pick_safest};
use crate::selfplay::battle::{run_sequential_games, SharedFileWriter};
use crate::selfplay::initialization::initialize_battle_state;
use crate::state::{MoveChoice, Pokemon, PokemonVolatileStatus, Side, SideConditions, State};
use clap::Parser;
use ndarray::Array2;
use serde_json::to_string_pretty;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use tch::Device;

struct IOData {
    state: State,
    instruction_list: Vec<Vec<Instruction>>,
    last_instructions_generated: Vec<StateInstructions>,
}

#[derive(Parser)]
struct Cli {
    #[clap(short, long, default_value = "")]
    state: String,

    #[clap(subcommand)]
    subcmd: Option<SubCommand>,
}

#[derive(Parser)]
enum SubCommand {
    Expectiminimax(Expectiminimax),
    IterativeDeepening(IterativeDeepening),
    MonteCarloTreeSearch(MonteCarloTreeSearch),
    MonteCarloTreeSearchEvo(MonteCarloTreeSearchEvo),
    MonteCarloTreeSearchPruned(MonteCarloTreeSearchPruned),
    MonteCarloTreeSearchPN(MonteCarloTreeSearchPN),
    MonteCarloTreeSearchMU(MonteCarloTreeSearchMU),
    CalculateDamage(CalculateDamage),
    GenerateInstructions(GenerateInstructions),
    GenerateObservation(GenerateObservation),
    GenerateEmbeddings(GenerateEmbeddings),
    SelfPlay(SelfPlay),
    AnalyzeMatchups(AnalyzeMatchups),
}

#[derive(Parser)]
pub struct SelfPlay {
    #[clap(long)]
    pub data_file: String,
    #[clap(long)]
    pub output_file: Option<String>,
    #[clap(long)]
    pub num_games: Option<usize>,
}

#[derive(Parser)]
pub struct AnalyzeMatchups {
    #[clap(short, long, required = true)]
    state: String,
}

#[derive(Parser)]
struct GenerateObservation {
    #[clap(short, long, required = true)]
    input_file: String,
    #[clap(short, long, required = true)]
    output_file: String,
}

#[derive(Parser)]
struct GenerateEmbeddings {}

#[derive(Parser)]
struct Expectiminimax {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = false)]
    ab_prune: bool,

    #[clap(short, long, default_value_t = 2)]
    depth: i8,
}

#[derive(Parser)]
struct IterativeDeepening {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = 5000)]
    time_to_search_ms: u64,
}

#[derive(Parser)]
struct MonteCarloTreeSearch {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = 5000)]
    time_to_search_ms: u64,
}

#[derive(Parser)]
struct MonteCarloTreeSearchMU {
    #[clap(short, long, required = false)]
    state: String,
    #[clap(short, long, default_value_t = 5000)]
    time_to_search_ms: u64,
}

#[derive(Parser)]
struct MonteCarloTreeSearchEvo {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = 5000)]
    time_to_search_ms: u64,
}

#[derive(Parser)]
struct MonteCarloTreeSearchPruned {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = 5000)]
    time_to_search_ms: u64,

    #[clap(short, long, required = true)]
    model_path: String,
}

#[derive(Parser)]
struct MonteCarloTreeSearchPN {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short, long, default_value_t = 3000)]
    time_to_search_ms: u64,

    #[clap(short, long, required = true)]
    model_path: String,

    #[clap(short, long, default_value_t = 5)]
    top_k: usize,
}

#[derive(Parser)]
struct CalculateDamage {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short = 'o', long, required = true)]
    side_one_move: String,

    #[clap(short = 't', long, required = true)]
    side_two_move: String,

    #[clap(short = 'm', long, required = false, default_value_t = false)]
    side_one_moves_first: bool,
}

#[derive(Parser)]
struct GenerateInstructions {
    #[clap(short, long, required = true)]
    state: String,

    #[clap(short = 'o', long, required = true)]
    side_one_move: String,

    #[clap(short = 't', long, required = true)]
    side_two_move: String,
}

impl Default for IOData {
    fn default() -> Self {
        IOData {
            state: State::default(),
            instruction_list: Vec::new(),
            last_instructions_generated: Vec::new(),
        }
    }
}

impl SideConditions {
    fn io_print(&self) -> String {
        let conditions = [
            ("aurora_veil", self.aurora_veil),
            ("crafty_shield", self.crafty_shield),
            ("healing_wish", self.healing_wish),
            ("light_screen", self.light_screen),
            ("lucky_chant", self.lucky_chant),
            ("lunar_dance", self.lunar_dance),
            ("mat_block", self.mat_block),
            ("mist", self.mist),
            ("protect", self.protect),
            ("quick_guard", self.quick_guard),
            ("reflect", self.reflect),
            ("safeguard", self.safeguard),
            ("spikes", self.spikes),
            ("stealth_rock", self.stealth_rock),
            ("sticky_web", self.sticky_web),
            ("tailwind", self.tailwind),
            ("toxic_count", self.toxic_count),
            ("toxic_spikes", self.toxic_spikes),
            ("wide_guard", self.wide_guard),
        ];

        let mut output = String::new();
        for (name, value) in conditions {
            if value != 0 {
                output.push_str(&format!("\n  {}: {}", name, value));
            }
        }
        if output.is_empty() {
            return "none".to_string();
        }
        output
    }
}

impl Side {
    fn io_print_boosts(&self) -> String {
        format!(
            "Attack:{}, Defense:{}, SpecialAttack:{}, SpecialDefense:{}, Speed:{}",
            self.attack_boost,
            self.defense_boost,
            self.special_attack_boost,
            self.special_defense_boost,
            self.speed_boost
        )
    }
    fn io_print(&self, available_choices: Vec<String>) -> String {
        let reserve = self
            .pokemon
            .into_iter()
            .map(|p| p.io_print_reserve())
            .collect::<Vec<String>>();
        format!(
            "\nActive:{}\nVolatiles: {:?}\nBoosts: {}\nSide Conditions: {}\nPokemon: {}\nAvailable Choices: {}",
            self.get_active_immutable().io_print_active(),
            self.volatile_statuses,
            self.io_print_boosts(),
            self.side_conditions.io_print(),
            reserve.join(", "),
            available_choices.join(", ")
        )
    }

    fn option_to_string(&self, option: &MoveChoice) -> String {
        match option {
            MoveChoice::MoveTera(index) => {
                format!("{}-tera", self.get_active_immutable().moves[index].id).to_lowercase()
            }
            MoveChoice::Move(index) => {
                format!("{}", self.get_active_immutable().moves[index].id).to_lowercase()
            }
            MoveChoice::Switch(index) => format!("{}", self.pokemon[*index].id).to_lowercase(),
            MoveChoice::None => "none".to_string(),
        }
    }

    pub fn string_to_movechoice(&self, s: &str) -> Option<MoveChoice> {
        let s = s.to_lowercase();
        if s == "none" {
            return Some(MoveChoice::None);
        }

        let mut pkmn_iter = self.pokemon.into_iter();
        while let Some(pkmn) = pkmn_iter.next() {
            if pkmn.id.to_string().to_lowercase() == s
                && pkmn_iter.pokemon_index != self.active_index
            {
                return Some(MoveChoice::Switch(pkmn_iter.pokemon_index));
            }
        }

        // check if s endswith `-tera`
        // if it does, find the move with the name and return MoveChoice::MoveTera
        // if it doesn't, find the move with the name and return MoveChoice::Move
        let mut move_iter = self.get_active_immutable().moves.into_iter();
        let mut move_name = s;
        if move_name.ends_with("-tera") {
            move_name = move_name[..move_name.len() - 5].to_string();
            while let Some(mv) = move_iter.next() {
                if format!("{:?}", mv.id).to_lowercase() == move_name {
                    return Some(MoveChoice::MoveTera(move_iter.pokemon_move_index));
                }
            }
        } else {
            while let Some(mv) = move_iter.next() {
                if format!("{:?}", mv.id).to_lowercase() == move_name {
                    return Some(MoveChoice::Move(move_iter.pokemon_move_index));
                }
            }
        }

        None
    }
}

impl Pokemon {
    fn io_print_reserve(&self) -> String {
        format!("{}:{}/{}", self.id, self.hp, self.maxhp)
    }
    fn io_print_active(&self) -> String {
        let moves: Vec<String> = self
            .moves
            .into_iter()
            .map(|m| format!("{:?}", m.id).to_lowercase())
            .filter(|x| x != "none")
            .collect();
        format!(
            "\n  Name: {}\n  HP: {}/{}\n  Status: {:?}\n  Ability: {:?}\n  Item: {:?}\n  Moves: {}",
            self.id,
            self.hp,
            self.maxhp,
            self.status,
            self.ability,
            self.item,
            moves.join(", ")
        )
    }
}

fn to_readable_name(variant: &str) -> String {
    // Remove common prefixes like HIDDEN_POWER or G_MAX
    let name = variant
        .trim_start_matches("HIDDENPOWER")
        .trim_start_matches("G_MAX")
        .trim_start_matches("MAX_");

    // Split by underscores
    let parts: Vec<&str> = name.split('_').collect();

    // Capitalize each part and join with spaces
    parts
        .iter()
        .map(|&part| {
            if part.is_empty() {
                String::new()
            } else if part.len() == 1 {
                part.to_uppercase()
            } else {
                let mut c = part.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            }
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<String>>()
        .join(" ")
}

fn generate_enum_mapping<T>(
    enum_values: &[(T, usize)],
    filename: &str,
    name_transform: fn(&str) -> String,
) -> std::io::Result<()>
where
    T: std::fmt::Debug,
{
    let mut map = HashMap::new();

    for (value, idx) in enum_values {
        // Convert enum variant to string
        let variant_str = format!("{:?}", value);

        // Apply name transformation
        let readable_name = name_transform(&variant_str);

        // Add to mapping (handle special case for index 0)
        map.insert(
            idx.to_string(),
            if *idx == 0 && readable_name.is_empty() {
                "None".to_string()
            } else {
                readable_name
            },
        );
    }

    // Convert to JSON
    let json_str = to_string_pretty(&map)?;

    // Write to file
    let mut file = File::create(filename)?;
    file.write_all(json_str.as_bytes())?;

    println!("Generated mapping file: {}", filename);
    Ok(())
}

fn write_embeddings_to_file<T>(
    embeddings: &[(T, ndarray::Array1<f32>)],
    filename: &str,
    id_to_u32: fn(T) -> u32,
) -> std::io::Result<()>
where
    T: Copy,
{
    let num_entries = embeddings.len();
    if num_entries == 0 {
        println!("No embeddings to write for {}", filename);
        return Ok(());
    }

    let embedding_dim = embeddings[0].1.len();

    // Create a matrix to store all embeddings
    let mut matrix = Array2::<f32>::zeros((num_entries, embedding_dim));
    let mut ids = Vec::with_capacity(num_entries);

    for (i, (id, embedding)) in embeddings.iter().enumerate() {
        matrix.row_mut(i).assign(embedding);
        ids.push(id_to_u32(*id));
    }

    // Save embeddings to file
    let mut file = File::create(filename)?;
    let shape = matrix.shape();

    // Write header: rows, cols
    file.write_all(&(shape[0] as u32).to_le_bytes())?;
    file.write_all(&(shape[1] as u32).to_le_bytes())?;

    // Write IDs
    for id in &ids {
        file.write_all(&id.to_le_bytes())?;
    }

    // Write flat data
    for &val in matrix.iter() {
        file.write_all(&val.to_le_bytes())?;
    }

    println!(
        "Saved {} {} embeddings with dimension {}",
        num_entries, filename, embedding_dim
    );
    Ok(())
}

fn generate_move_embeddings() -> std::io::Result<()> {
    // Use the existing MOVES hashmap from choices.rs
    let move_embeddings = embedding::create_all_move_embeddings(&*choices::MOVES);

    // Convert HashMap entries to a Vec of tuples for sorted writing
    let mut sorted_moves: Vec<_> = move_embeddings.into_iter().collect();
    sorted_moves.sort_by_key(|(choice_id, _)| *choice_id as u16);

    // Create a vector of (enum, index) pairs for name mapping
    let named_moves: Vec<_> = sorted_moves
        .iter()
        .enumerate()
        .map(|(idx, (choice_id, _))| (*choice_id, idx))
        .collect();

    // Generate name mapping file
    generate_enum_mapping(&named_moves, "move_names.json", to_readable_name)?;

    write_embeddings_to_file(&sorted_moves, "move_embeddings.bin", |choice| {
        choice as u16 as u32
    })
}

fn generate_ability_embeddings() -> std::io::Result<()> {
    let ability_embeddings = embedding::create_all_ability_embeddings();

    // Create a vector of (enum, index) pairs for name mapping
    let named_abilities: Vec<_> = ability_embeddings
        .iter()
        .enumerate()
        .map(|(idx, (ability, _))| (*ability, idx))
        .collect();

    // Generate name mapping file
    generate_enum_mapping(&named_abilities, "ability_names.json", to_readable_name)?;

    write_embeddings_to_file(&ability_embeddings, "ability_embeddings.bin", |ability| {
        ability as u16 as u32
    })
}

fn generate_item_embeddings() -> std::io::Result<()> {
    let item_embeddings = embedding::create_all_item_embeddings();

    // Create a vector of (enum, index) pairs for name mapping
    let named_items: Vec<_> = item_embeddings
        .iter()
        .enumerate()
        .map(|(idx, (item, _))| (*item, idx))
        .collect();

    // Generate name mapping file
    generate_enum_mapping(&named_items, "item_names.json", to_readable_name)?;

    write_embeddings_to_file(&item_embeddings, "item_embeddings.bin", |item| {
        item as u16 as u32
    })
}

pub fn io_get_all_options(state: &State) -> (Vec<MoveChoice>, Vec<MoveChoice>) {
    if state.team_preview {
        let mut s1_options = Vec::with_capacity(6);
        let mut s2_options = Vec::with_capacity(6);

        let mut pkmn_iter = state.side_one.pokemon.into_iter();
        while let Some(_) = pkmn_iter.next() {
            if state.side_one.pokemon[pkmn_iter.pokemon_index].hp > 0 {
                s1_options.push(MoveChoice::Switch(pkmn_iter.pokemon_index));
            }
        }
        let mut pkmn_iter = state.side_two.pokemon.into_iter();
        while let Some(_) = pkmn_iter.next() {
            if state.side_two.pokemon[pkmn_iter.pokemon_index].hp > 0 {
                s2_options.push(MoveChoice::Switch(pkmn_iter.pokemon_index));
            }
        }
        return (s1_options, s2_options);
    }

    let (mut s1_options, mut s2_options) = state.get_all_options();

    if state.side_one.force_trapped {
        s1_options.retain(|x| match x {
            MoveChoice::Move(_) | MoveChoice::MoveTera(_) => true,
            MoveChoice::Switch(_) => false,
            MoveChoice::None => true,
        });
    }
    if state.side_one.slow_uturn_move {
        s1_options.clear();
        let encored = state
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::ENCORE);
        state.side_one.get_active_immutable().add_available_moves(
            &mut s1_options,
            &state.side_one.last_used_move,
            encored,
            state.side_one.can_use_tera(),
        );
    }

    if state.side_two.force_trapped {
        s2_options.retain(|x| match x {
            MoveChoice::Move(_) | MoveChoice::MoveTera(_) => true,
            MoveChoice::Switch(_) => false,
            MoveChoice::None => true,
        });
    }
    if state.side_two.slow_uturn_move {
        s2_options.clear();
        let encored = state
            .side_two
            .volatile_statuses
            .contains(&PokemonVolatileStatus::ENCORE);
        state.side_two.get_active_immutable().add_available_moves(
            &mut s2_options,
            &state.side_two.last_used_move,
            encored,
            state.side_two.can_use_tera(),
        );
    }

    if s1_options.len() == 0 {
        s1_options.push(MoveChoice::None);
    }
    if s2_options.len() == 0 {
        s2_options.push(MoveChoice::None);
    }

    (s1_options, s2_options)
}

fn pprint_expectiminimax_result(
    result: &Vec<f32>,
    s1_options: &Vec<MoveChoice>,
    s2_options: &Vec<MoveChoice>,
    safest_choice: &(usize, f32),
    state: &State,
) {
    let s1_len = s1_options.len();
    let s2_len = s2_options.len();

    print!("{: <12}", " ");

    for s2_move in s2_options.iter() {
        match s2_move {
            MoveChoice::MoveTera(m) => {
                let s2_move_str =
                    format!("{}-tera", state.side_two.get_active_immutable().moves[m].id);
                print!("{: >12}", s2_move_str.to_lowercase());
            }
            MoveChoice::Move(m) => {
                let s2_move_str = format!("{}", state.side_two.get_active_immutable().moves[m].id);
                print!("{: >12}", s2_move_str.to_lowercase());
            }
            MoveChoice::Switch(s) => {
                let s2_move_str = format!(
                    "{}",
                    state.side_two.pokemon[*s].id.to_string().to_lowercase()
                );
                print!("{: >12}", s2_move_str);
            }
            MoveChoice::None => {}
        }
    }
    print!("\n");

    for i in 0..s1_len {
        let s1_move_str = s1_options[i];
        match s1_move_str {
            MoveChoice::MoveTera(m) => {
                let move_id = format!(
                    "{}-tera",
                    state.side_one.get_active_immutable().moves[&m].id
                );
                print!("{:<12}", move_id.to_string().to_lowercase());
            }
            MoveChoice::Move(m) => {
                let move_id = state.side_one.get_active_immutable().moves[&m].id;
                print!("{:<12}", move_id.to_string().to_lowercase());
            }
            MoveChoice::Switch(s) => {
                let pkmn_id = &state.side_one.pokemon[s].id;
                print!("{:<12}", pkmn_id.to_string().to_lowercase());
            }
            MoveChoice::None => {}
        }
        for j in 0..s2_len {
            let index = i * s2_len + j;
            print!("{number:>11.2} ", number = result[index]);
        }
        print!("\n");
    }
    match s1_options[safest_choice.0] {
        MoveChoice::MoveTera(m) => {
            let move_id = format!(
                "{}-tera",
                state.side_one.get_active_immutable().moves[&m].id
            );
            print!(
                "\nSafest Choice: {}, {}\n",
                move_id.to_string().to_lowercase(),
                safest_choice.1
            );
        }
        MoveChoice::Move(m) => {
            let move_id = state.side_one.get_active_immutable().moves[&m].id;
            print!(
                "\nSafest Choice: {}, {}\n",
                move_id.to_string().to_lowercase(),
                safest_choice.1
            );
        }
        MoveChoice::Switch(s) => {
            let pkmn_id = &state.side_one.pokemon[s].id;
            print!(
                "\nSafest Choice: Switch {}, {}\n",
                pkmn_id.to_string().to_lowercase(),
                safest_choice.1
            );
        }
        MoveChoice::None => println!("No Move"),
    }
}

fn pprint_mcts_result(state: &State, result: MctsResult) {
    println!("\nTotal Iterations: {}\n", result.iteration_count);
    println!("Maximum Depth: {}", result.max_depth);
    println!("Side One:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits"
    );
    for x in result.s1.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}",
            get_move_id_from_movechoice(&state.side_one, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0
        );
    }

    println!("Side Two:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits"
    );
    for x in result.s2.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}",
            get_move_id_from_movechoice(&state.side_two, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0
        );
    }
}

fn pprint_mcts_result_pruned(state: &State, result: MctsResultPruned) {
    println!("\nTotal Iterations: {}\n", result.iteration_count);
    println!("Maximum Depth: {}", result.max_depth);
    println!("Side One:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}{:>12}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits", "Prior Prob"
    );
    for x in result.s1.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}{:>12.3}",
            get_move_id_from_movechoice(&state.side_one, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0,
            x.prior_prob
        );
    }

    println!("Side Two:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}{:>12}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits", "Prior Prob"
    );
    for x in result.s2.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}{:>12.3}",
            get_move_id_from_movechoice(&state.side_two, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0,
            x.prior_prob
        );
    }
}

fn pprint_mcts_result_pn(state: &State, result: MctsResultPN) {
    println!("\nTotal Iterations: {}\n", result.iteration_count);
    println!("Maximum Depth: {}", result.max_depth);
    println!("Side One:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}{:>12}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits", "Prior Prob"
    );
    for x in result.s1.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}{:>12.3}",
            get_move_id_from_movechoice(&state.side_one, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0,
            x.prior_prob
        );
    }

    println!("Side Two:");
    println!(
        "\t{:<25}{:>12}{:>12}{:>10}{:>10}{:>12}",
        "Move", "Total Score", "Avg Score", "Visits", "% Visits", "Prior Prob"
    );
    for x in result.s2.iter() {
        println!(
            "\t{:<25}{:>12.2}{:>12.2}{:>10}{:>10.2}{:>12.3}",
            get_move_id_from_movechoice(&state.side_two, &x.move_choice),
            x.total_score,
            x.total_score / x.visits as f32,
            x.visits,
            (x.visits as f32 / result.iteration_count as f32) * 100.0,
            x.prior_prob
        );
    }
}

fn pprint_state_instruction_vector(instructions: &Vec<StateInstructions>) {
    for (i, instruction) in instructions.iter().enumerate() {
        println!("Index: {}", i);
        println!("StateInstruction: {:?}", instruction);
    }
}

fn get_move_id_from_movechoice(side: &Side, move_choice: &MoveChoice) -> String {
    match move_choice {
        MoveChoice::MoveTera(index) => {
            format!("{}-tera", side.get_active_immutable().moves[&index].id).to_lowercase()
        }
        MoveChoice::Move(index) => {
            format!("{}", side.get_active_immutable().moves[&index].id).to_lowercase()
        }
        MoveChoice::Switch(index) => format!("switch {}", side.pokemon[*index].id).to_lowercase(),
        MoveChoice::None => "No Move".to_string(),
    }
}

fn print_subcommand_result(
    result: &Vec<f32>,
    side_one_options: &Vec<MoveChoice>,
    side_two_options: &Vec<MoveChoice>,
    state: &State,
) {
    let safest = pick_safest(&result, side_one_options.len(), side_two_options.len());
    let move_choice = side_one_options[safest.0];

    let joined_side_one_options = side_one_options
        .iter()
        .map(|x| format!("{}", get_move_id_from_movechoice(&state.side_one, x)))
        .collect::<Vec<String>>()
        .join(",");
    println!("side one options: {}", joined_side_one_options);

    let joined_side_two_options = side_two_options
        .iter()
        .map(|x| format!("{}", get_move_id_from_movechoice(&state.side_two, x)))
        .collect::<Vec<String>>()
        .join(",");
    println!("side two options: {}", joined_side_two_options);

    let joined = result
        .iter()
        .map(|x| format!("{:.2}", x))
        .collect::<Vec<String>>()
        .join(",");
    println!("matrix: {}", joined);
    match move_choice {
        MoveChoice::MoveTera(_) => {
            println!(
                "choice: {}-tera",
                get_move_id_from_movechoice(&state.side_one, &move_choice)
            );
        }
        MoveChoice::Move(_) => {
            println!(
                "choice: {}",
                get_move_id_from_movechoice(&state.side_one, &move_choice)
            );
        }
        MoveChoice::Switch(_) => {
            println!(
                "choice: switch {}",
                get_move_id_from_movechoice(&state.side_one, &move_choice)
            );
        }
        MoveChoice::None => {
            println!("no move");
        }
    }
    println!("evaluation: {}", safest.1);
}

pub fn main() {
    let args = Cli::parse();
    let mut io_data = IOData::default();

    if args.state != "" {
        let state = State::deserialize(args.state.as_str());
        io_data.state = state;
    }

    let result;
    let mut state;
    let mut side_one_options;
    let mut side_two_options;
    match args.subcmd {
        None => {
            command_loop(io_data);
            exit(0);
        }
        Some(subcmd) => match subcmd {
            SubCommand::SelfPlay(args) => {
                let data_dir = PathBuf::from("data");
                let random_teams_path = data_dir.join("random_teams.json");
                let pokedex_path = data_dir.join("pokedex.json");
                let moves_path = data_dir.join("moves.json");
                let random_teams = fs::read_to_string(random_teams_path).unwrap();
                let pokedex = fs::read_to_string(pokedex_path).unwrap();
                let movedex = fs::read_to_string(moves_path).unwrap();

                let data_path = std::env::current_dir().unwrap().join(args.data_file);
                println!("Writing training data to: {}", data_path.display());

                // Create log directory if logging is requested
                let log_dir = args.output_file.map(|_| {
                    let log_dir = std::env::current_dir().unwrap().join("logs").join(format!(
                        "selfplay_{}",
                        chrono::Local::now().format("%Y%m%d_%H%M%S")
                    ));
                    fs::create_dir_all(&log_dir).unwrap();
                    println!("Writing battle logs to: {}", log_dir.display());
                    log_dir
                });

                // Create shared writer
                let writer = Arc::new(SharedFileWriter::new(data_path).unwrap());

                run_sequential_games(
                    args.num_games.unwrap_or(1),
                    writer,
                    &random_teams,
                    &pokedex,
                    &movedex,
                    log_dir,
                )
                .unwrap();
            }

            SubCommand::AnalyzeMatchups(analyze_matchups) => {
                state = State::deserialize(analyze_matchups.state.as_str());
                // Initialize cache and visualizer
                let cache = initialize_team_matchup_cache(&state);
                let mut visualizer = MatchupVisualizer::new(cache);

                // Analyze all matchups with detailed reasoning
                visualizer.analyze_all_matchups(&state);

                // Print the matchup matrix
                visualizer.print_matchup_matrix(&state);

                // Generate HTML visualization
                match visualizer.generate_html_visualization(&state) {
                    Ok(_) => println!(
                        "HTML visualization has been generated in the 'matchup_analysis' directory"
                    ),
                    Err(e) => println!("Failed to generate HTML visualization: {}", e),
                };

                println!(
                    "\nCommand: 'matchup-detail <s1_idx> <s2_idx>' for specific matchup details"
                );
                println!("Example: 'matchup-detail 0 2' to see details of first Pokémon vs third opponent Pokémon");
            }

            SubCommand::GenerateObservation(args) => {
                let state_string =
                    std::fs::read_to_string(&args.input_file).expect("Failed to read input file");
                let state_string = state_string.trim();
                generate_observation_output(state_string, args.output_file.trim())
                    .expect("Failed to write inspection file");
            }
            SubCommand::GenerateEmbeddings(_) => {
                println!("Generating Pokemon embeddings...");

                // Generate ability embeddings
                if let Err(e) = generate_ability_embeddings() {
                    eprintln!("Error generating ability embeddings: {}", e);
                }

                // Generate item embeddings
                if let Err(e) = generate_item_embeddings() {
                    eprintln!("Error generating item embeddings: {}", e);
                }

                // Generate move embeddings
                if let Err(e) = generate_move_embeddings() {
                    eprintln!("Error generating move embeddings: {}", e);
                }

                println!("Embeddings generation complete!");
            }
            SubCommand::Expectiminimax(expectiminimax) => {
                state = State::deserialize(expectiminimax.state.as_str());
                (side_one_options, side_two_options) = io_get_all_options(&state);
                result = expectiminimax_search(
                    &mut state,
                    expectiminimax.depth,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    expectiminimax.ab_prune,
                    &Arc::new(Mutex::new(true)),
                );
                print_subcommand_result(&result, &side_one_options, &side_two_options, &state);
            }
            SubCommand::IterativeDeepening(iterative_deepending) => {
                state = State::deserialize(iterative_deepending.state.as_str());
                (side_one_options, side_two_options) = io_get_all_options(&state);
                (side_one_options, side_two_options, result, _) = iterative_deepen_expectiminimax(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(iterative_deepending.time_to_search_ms),
                );
                print_subcommand_result(&result, &side_one_options, &side_two_options, &state);
            }
            SubCommand::MonteCarloTreeSearch(mcts) => {
                state = State::deserialize(mcts.state.as_str());
                (side_one_options, side_two_options) = io_get_all_options(&state);
                let result = perform_mcts(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(mcts.time_to_search_ms),
                );
                pprint_mcts_result(&state, result);
            }

            SubCommand::MonteCarloTreeSearchMU(mcts_evo) => {
                // if state is provided, use it
                let mut state = if mcts_evo.state != "" {
                    State::deserialize(mcts_evo.state.as_str())
                } else {
                    // if not, read from data files
                    let data_dir = PathBuf::from("data");
                    let random_teams = match fs::read_to_string(data_dir.join("random_teams.json"))
                    {
                        Ok(content) => content,
                        Err(err) => {
                            eprintln!("Failed to read file: {}", err);
                            return;
                        }
                    };
                    let pokedex = match fs::read_to_string(data_dir.join("pokedex.json")) {
                        Ok(content) => content,
                        Err(err) => {
                            eprintln!("Failed to read file: {}", err);
                            return;
                        }
                    };
                    let movedex = match fs::read_to_string(data_dir.join("moves.json")) {
                        Ok(content) => content,
                        Err(err) => {
                            eprintln!("Failed to read file: {}", err);
                            return;
                        }
                    };
                    initialize_battle_state(&random_teams, &pokedex, &movedex)
                };
                println!("{}", state.serialize());
                let mut matchup_cache = initialize_team_matchup_cache(&state);

                analyze_matchup_cache(&state, &mut matchup_cache);

                (side_one_options, side_two_options) = io_get_all_options(&state);

                let result = perform_mcts_with_team_matchups(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(mcts_evo.time_to_search_ms),
                    &mut matchup_cache,
                );
                pprint_mcts_result(&state, result);
                // Print cache statistics
                let (hits, misses, total, hit_rate) = matchup_cache.get_stats();
                println!("\nCache Statistics:");
                println!("  Total queries: {}", total);
                println!("  Cache hits: {} ({:.2}%)", hits, hit_rate);
                println!("  Cache misses: {} ({:.2}%)", misses, 100.0 - hit_rate);
                println!("  Cache size: {} entries", matchup_cache.cache_size());
                let (
                    complete,
                    partial,
                    fallbacks,
                    total,
                    complete_rate,
                    partial_rate,
                    fallback_rate,
                ) = matchup_cache.get_evaluation_stats();

                println!("\nEvaluation Statistics:");
                println!("  Total evaluations: {}", total);
                println!(
                    "  Complete evaluations: {} ({:.2}%)",
                    complete, complete_rate
                );
                println!("  Partial evaluations: {} ({:.2}%)", partial, partial_rate);
                println!("  Fallbacks: {} ({:.2}%)", fallbacks, fallback_rate);
                println!(
                    "  Strategic value mean: {}",
                    matchup_cache.team_strategic_value_mean
                );
                println!("  Base value mean: {}", matchup_cache.team_base_value_mean);
            }

            SubCommand::MonteCarloTreeSearchEvo(mcts_evo) => {
                state = State::deserialize(mcts_evo.state.as_str());
                (side_one_options, side_two_options) = io_get_all_options(&state);
                let result = perform_mcts(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(mcts_evo.time_to_search_ms),
                );
                pprint_mcts_result(&state, result);
            }

            SubCommand::MonteCarloTreeSearchPruned(mcts_pruned) => {
                state = State::deserialize(mcts_pruned.state.as_str());
                let device = Device::Cpu; // or Device::Cuda(0) for GPU
                let model = match JointNetwork::new(&mcts_pruned.model_path, device) {
                    Ok(model) => Arc::new(model),
                    Err(e) => panic!("Failed to load model: {}", e),
                };
                (side_one_options, side_two_options) = io_get_all_options(&state);
                let result = perform_mcts_pruned_batched(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(mcts_pruned.time_to_search_ms),
                    model.clone(),
                );
                pprint_mcts_result_pruned(&state, result);
            }

            SubCommand::MonteCarloTreeSearchPN(mcts_pn) => {
                state = State::deserialize(mcts_pn.state.as_str());
                let device = Device::Cpu; // or Device::Cuda(0) for GPU
                let model = match PolicyNetwork::new(&mcts_pn.model_path, device) {
                    Ok(model) => Arc::new(model),
                    Err(e) => panic!("Failed to load model: {}", e),
                };
                (side_one_options, side_two_options) = io_get_all_options(&state);
                let result = perform_mcts_pn_batched(
                    &mut state,
                    side_one_options.clone(),
                    side_two_options.clone(),
                    std::time::Duration::from_millis(mcts_pn.time_to_search_ms),
                    model.clone(),
                    mcts_pn.top_k,
                );
                pprint_mcts_result_pn(&state, result);
            }

            SubCommand::CalculateDamage(calculate_damage) => {
                state = State::deserialize(calculate_damage.state.as_str());
                let mut s1_choice = MOVES
                    .get(&Choices::from_str(calculate_damage.side_one_move.as_str()).unwrap())
                    .unwrap()
                    .to_owned();
                let mut s2_choice = MOVES
                    .get(&Choices::from_str(calculate_damage.side_two_move.as_str()).unwrap())
                    .unwrap()
                    .to_owned();
                let s1_moves_first = calculate_damage.side_one_moves_first;
                if calculate_damage.side_one_move == "switch" {
                    s1_choice.category = MoveCategory::Switch
                }
                if calculate_damage.side_two_move == "switch" {
                    s2_choice.category = MoveCategory::Switch
                }
                calculate_damage_io(&state, s1_choice, s2_choice, s1_moves_first);
            }
            SubCommand::GenerateInstructions(generate_instructions) => {
                state = State::deserialize(generate_instructions.state.as_str());
                let (s1_movechoice, s2_movechoice);
                match state
                    .side_one
                    .string_to_movechoice(generate_instructions.side_one_move.as_str())
                {
                    None => {
                        println!(
                            "Invalid move choice for side one: {}",
                            generate_instructions.side_one_move
                        );
                        exit(1);
                    }
                    Some(v) => s1_movechoice = v,
                }
                match state
                    .side_two
                    .string_to_movechoice(generate_instructions.side_two_move.as_str())
                {
                    None => {
                        println!(
                            "Invalid move choice for side two: {}",
                            generate_instructions.side_two_move
                        );
                        exit(1);
                    }
                    Some(v) => s2_movechoice = v,
                }
                let instructions = generate_instructions_from_move_pair(
                    &mut state,
                    &s1_movechoice,
                    &s2_movechoice,
                    true,
                );
                pprint_state_instruction_vector(&instructions);
            }
        },
    }

    exit(0);
}

fn calculate_damage_io(
    state: &State,
    s1_choice: Choice,
    s2_choice: Choice,
    side_one_moves_first: bool,
) {
    let (damages_dealt_s1, damages_dealt_s2) =
        calculate_both_damage_rolls(state, s1_choice, s2_choice, side_one_moves_first);

    for dmg in [damages_dealt_s1, damages_dealt_s2] {
        match dmg {
            Some(damages_vec) => {
                let joined = damages_vec
                    .iter()
                    .map(|x| format!("{:?}", x))
                    .collect::<Vec<String>>()
                    .join(",");
                println!("Damage Rolls: {}", joined);
            }
            None => {
                println!("Damage Rolls: 0");
            }
        }
    }
}

fn command_loop(mut io_data: IOData) {
    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {}
            Err(error) => {
                println!("Error reading input: {}", error);
                continue;
            }
        }
        let mut parts = input.trim().split_whitespace();
        let command = parts.next().unwrap_or("");
        let mut args = parts;

        match command {
            "state" | "s" => {
                let state_string;
                match args.next() {
                    Some(s) => {
                        state_string = s;
                        let state = State::deserialize(state_string);
                        io_data.state = state;
                        println!("state initialized");
                    }
                    None => {
                        println!("Expected state string");
                    }
                }
                println!("{:?}", io_data.state);
            }
            "serialize" | "ser" => {
                println!("{}", io_data.state.serialize());
            }
            "matchup" | "m" => {
                let (side_one_options, side_two_options) = io_get_all_options(&io_data.state);

                let mut side_one_choices = vec![];
                for option in side_one_options {
                    side_one_choices.push(
                        format!("{:?}", io_data.state.side_one.option_to_string(&option))
                            .to_lowercase(),
                    );
                }
                let mut side_two_choices = vec![];
                for option in side_two_options {
                    side_two_choices.push(
                        format!("{:?}", io_data.state.side_two.option_to_string(&option))
                            .to_lowercase(),
                    );
                }
                println!(
                    "SideOne {}\n\nvs\n\nSideTwo {}\n\nState:\n  Weather: {:?},{}\n  Terrain: {:?},{}\n  TrickRoom: {},{}",
                    io_data.state.side_one.io_print(side_one_choices),
                    io_data.state.side_two.io_print(side_two_choices),
                    io_data.state.weather.weather_type,
                    io_data.state.weather.turns_remaining,
                    io_data.state.terrain.terrain_type,
                    io_data.state.terrain.turns_remaining,
                    io_data.state.trick_room.active,
                    io_data.state.trick_room.turns_remaining
                );
            }
            "generate-instructions" | "g" => {
                let (s1_move, s2_move);
                match args.next() {
                    Some(s) => match io_data.state.side_one.string_to_movechoice(s) {
                        Some(m) => {
                            s1_move = m;
                        }
                        None => {
                            println!("Invalid move choice for side one: {}", s);
                            continue;
                        }
                    },
                    None => {
                        println!("Usage: generate-instructions <side-1 move> <side-2 move>");
                        continue;
                    }
                }
                match args.next() {
                    Some(s) => match io_data.state.side_two.string_to_movechoice(s) {
                        Some(m) => {
                            s2_move = m;
                        }
                        None => {
                            println!("Invalid move choice for side two: {}", s);
                            continue;
                        }
                    },
                    None => {
                        println!("Usage: generate-instructions <side-1 choice> <side-2 choice>");
                        continue;
                    }
                }
                let instructions = generate_instructions_from_move_pair(
                    &mut io_data.state,
                    &s1_move,
                    &s2_move,
                    true,
                );
                pprint_state_instruction_vector(&instructions);
                io_data.last_instructions_generated = instructions;
            }
            "calculate-damage" | "d" => {
                let (mut s1_choice, mut s2_choice);
                match args.next() {
                    Some(s) => {
                        s1_choice = MOVES
                            .get(&Choices::from_str(s).unwrap())
                            .unwrap()
                            .to_owned();
                        if s == "switch" {
                            s1_choice.category = MoveCategory::Switch
                        }
                    }
                    None => {
                        println!("Usage: calculate-damage <side-1 move> <side-2 move> <side-1-moves-first>");
                        continue;
                    }
                }
                match args.next() {
                    Some(s) => {
                        s2_choice = MOVES
                            .get(&Choices::from_str(s).unwrap())
                            .unwrap()
                            .to_owned();
                        if s == "switch" {
                            s2_choice.category = MoveCategory::Switch
                        }
                    }
                    None => {
                        println!("Usage: calculate-damage <side-1 move> <side-2 move> <side-1-moves-first>");
                        continue;
                    }
                }
                let s1_moves_first: bool;
                match args.next() {
                    Some(s) => {
                        s1_moves_first = s.parse::<bool>().unwrap();
                    }
                    None => {
                        println!("Usage: calculate-damage <side-1 move> <side-2 move> <side-1-moves-first>");
                        continue;
                    }
                }
                calculate_damage_io(&io_data.state, s1_choice, s2_choice, s1_moves_first);
            }
            "instructions" | "i" => {
                println!("{:?}", io_data.last_instructions_generated);
            }
            "evaluate" | "ev" => {
                println!("Evaluation: {}", evaluate(&io_data.state));
            }
            "iterative-deepening" | "id" => match args.next() {
                Some(s) => {
                    let max_time_ms = s.parse::<u64>().unwrap();
                    let (side_one_options, side_two_options) = io_get_all_options(&io_data.state);

                    let start_time = std::time::Instant::now();
                    let (s1_moves, s2_moves, result, depth_searched) =
                        iterative_deepen_expectiminimax(
                            &mut io_data.state,
                            side_one_options.clone(),
                            side_two_options.clone(),
                            std::time::Duration::from_millis(max_time_ms),
                        );
                    let elapsed = start_time.elapsed();

                    let safest_choice = pick_safest(&result, s1_moves.len(), s2_moves.len());

                    pprint_expectiminimax_result(
                        &result,
                        &s1_moves,
                        &s2_moves,
                        &safest_choice,
                        &io_data.state,
                    );
                    println!("Took: {:?}", elapsed);
                    println!("Depth Searched: {}", depth_searched);
                }
                None => {
                    println!("Usage: iterative-deepening <timeout_ms>");
                    continue;
                }
            },
            "monte-carlo-tree-search" | "mcts" => match args.next() {
                Some(s) => {
                    let max_time_ms = s.parse::<u64>().unwrap();
                    let (side_one_options, side_two_options) = io_get_all_options(&io_data.state);

                    let start_time = std::time::Instant::now();
                    let result = perform_mcts(
                        &mut io_data.state,
                        side_one_options.clone(),
                        side_two_options.clone(),
                        std::time::Duration::from_millis(max_time_ms),
                    );
                    let elapsed = start_time.elapsed();
                    pprint_mcts_result(&io_data.state, result);

                    println!("\nTook: {:?}", elapsed);
                }
                None => {
                    println!("Usage: monte-carlo-tree-search <timeout_ms>");
                    continue;
                }
            },
            "apply" | "a" => match args.next() {
                Some(s) => {
                    let index = s.parse::<usize>().unwrap();
                    let instructions = io_data.last_instructions_generated.remove(index);
                    io_data
                        .state
                        .apply_instructions(&instructions.instruction_list);
                    io_data.instruction_list.push(instructions.instruction_list);
                    io_data.last_instructions_generated = Vec::new();
                    println!("Applied instructions at index {}", index)
                }
                None => {
                    println!("Usage: apply <instruction index>");
                    continue;
                }
            },
            "pop" | "p" => {
                if io_data.instruction_list.is_empty() {
                    println!("No instructions to pop");
                    continue;
                }
                let instructions = io_data.instruction_list.pop().unwrap();
                io_data.state.reverse_instructions(&instructions);
                println!("Popped last applied instructions");
            }
            "pop-all" | "pa" => {
                for i in io_data.instruction_list.iter().rev() {
                    io_data.state.reverse_instructions(i);
                }
                io_data.instruction_list.clear();
                println!("Popped all applied instructions");
            }
            "expectiminimax" | "e" => match args.next() {
                Some(s) => {
                    let mut ab_prune = false;
                    match args.next() {
                        Some(s) => ab_prune = s.parse::<bool>().unwrap(),
                        None => {}
                    }
                    let depth = s.parse::<i8>().unwrap();
                    let (side_one_options, side_two_options) = io_get_all_options(&io_data.state);
                    let start_time = std::time::Instant::now();
                    let result = expectiminimax_search(
                        &mut io_data.state,
                        depth,
                        side_one_options.clone(),
                        side_two_options.clone(),
                        ab_prune,
                        &Arc::new(Mutex::new(true)),
                    );
                    let elapsed = start_time.elapsed();

                    let safest_choice =
                        pick_safest(&result, side_one_options.len(), side_two_options.len());
                    pprint_expectiminimax_result(
                        &result,
                        &side_one_options,
                        &side_two_options,
                        &safest_choice,
                        &io_data.state,
                    );
                    println!("\nTook: {:?}", elapsed);
                }
                None => {
                    println!("Usage: expectiminimax <depth> <ab_prune=false>");
                    continue;
                }
            },
            "" => {
                continue;
            }
            "exit" | "quit" | "q" => {
                break;
            }
            command => {
                println!("Unknown command: {}", command);
            }
        }
    }
}
