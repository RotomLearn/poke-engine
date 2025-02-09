use crate::abilities::Abilities;
use crate::choices::Choices;
use crate::items::Items;
use crate::pokemon::PokemonName;
use crate::state::LastUsedMove;
use crate::state::{
    Move, Pokemon, PokemonIndex, PokemonMoves, PokemonNature, PokemonStatus, PokemonType, Side,
    SideConditions, SidePokemon, State, StateTerrain, StateTrickRoom, StateWeather, Terrain,
    Weather,
};
use deunicode::deunicode;
use serde::Deserialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::str::FromStr;

#[derive(Deserialize)]
struct PackedTeamEntry {
    id: i32,
    format: String,
    #[serde(rename = "packedTeam")]
    packed_team: String,
}

#[derive(Deserialize)]
struct PokemonDexEntry {
    name: String,
    types: Vec<String>,
    #[serde(rename = "baseStats")]
    base_stats: BaseStats,
    weightkg: f32,
}

#[derive(Deserialize)]
struct BaseStats {
    hp: u16,
    attack: u16,
    defense: u16,
    #[serde(rename = "special-attack")]
    special_attack: u16,
    #[serde(rename = "special-defense")]
    special_defense: u16,
    speed: u16,
}

fn normalize_name(name: &str) -> String {
    deunicode(name) // Handle unicode characters
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
        .replace("'", "")
        .replace("%", "")
        .replace("*", "")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .trim()
        .to_lowercase()
}

#[derive(Debug)]
struct PackedPokemon {
    nickname: String,
    species: Option<String>, // None if identical to nickname
    item: String,
    ability: String,
    moves: Vec<String>,
    nature: String,
    evs: Vec<u8>,
    gender: String,
    ivs: Vec<u8>,
    shiny: bool,
    level: u8,
    tera_type: String,
}

#[derive(Deserialize)]
struct MoveDexEntry {
    pp: i8,
    #[serde(flatten)]
    _other: HashMap<String, serde_json::Value>,
}

fn parse_packed_team(packed_team: &str) -> Vec<PackedPokemon> {
    packed_team
        .split(']')
        .filter(|s| !s.is_empty())
        .map(|pokemon_str| {
            let parts: Vec<&str> = pokemon_str.split('|').collect();

            // Parse EVs
            let evs = if parts[6].is_empty() {
                vec![0, 0, 0, 0, 0, 0]
            } else {
                let mut ev_list = parts[6]
                    .split(',')
                    .map(|s| s.parse::<u8>().unwrap_or(0))
                    .collect::<Vec<_>>();
                while ev_list.len() < 6 {
                    ev_list.push(0);
                }
                ev_list
            };

            // Parse IVs
            let ivs = if parts.len() > 8 && !parts[8].is_empty() {
                let mut iv_list = parts[8]
                    .split(',')
                    .map(|s| s.parse::<u8>().unwrap_or(31))
                    .collect::<Vec<_>>();
                while iv_list.len() < 6 {
                    iv_list.push(31);
                }
                iv_list
            } else {
                vec![31, 31, 31, 31, 31, 31]
            };

            // Get level
            let level = if parts.len() > 10 && !parts[10].is_empty() {
                parts[10].parse().unwrap_or(100)
            } else {
                100
            };

            // Get tera type from last comma-separated value in the last part
            let tera_type = if parts.last().unwrap().contains(',') {
                parts
                    .last()
                    .unwrap()
                    .split(',')
                    .last()
                    .unwrap_or("")
                    .to_string()
            } else {
                "".to_string()
            };

            PackedPokemon {
                nickname: normalize_name(parts[0]),
                species: if parts[1].is_empty() {
                    None
                } else {
                    Some(normalize_name(parts[1]))
                },
                item: normalize_name(parts[2]),
                ability: normalize_name(parts[3]),
                moves: parts[4].split(',').map(normalize_name).collect(),
                nature: if parts[5].is_empty() {
                    "serious".to_string()
                } else {
                    normalize_name(parts[5])
                },
                evs,
                gender: parts[7].to_string(),
                ivs,
                shiny: parts.len() > 9 && parts[9] == "S",
                level,
                tera_type,
            }
        })
        .collect()
}

fn create_pokemon(
    packed: &PackedPokemon,
    pokedex: &HashMap<String, PokemonDexEntry>,
    movedex: &HashMap<String, MoveDexEntry>,
) -> Pokemon {
    // Previous pokedex lookup logic remains the same
    let pokemon_name = packed.species.as_ref().unwrap_or(&packed.nickname);
    let dex_entry = pokedex
        .get(pokemon_name)
        .unwrap_or_else(|| panic!("Pokemon not found in pokedex: {}", pokemon_name));

    // Type conversion remains the same
    let type1 =
        PokemonType::from_str(&normalize_name(&dex_entry.types[0])).unwrap_or(PokemonType::NORMAL);
    let type2 = if dex_entry.types.len() > 1 {
        PokemonType::from_str(&normalize_name(&dex_entry.types[1])).unwrap_or(PokemonType::TYPELESS)
    } else {
        PokemonType::TYPELESS
    };

    let mut moves = PokemonMoves {
        m0: Move::default(),
        m1: Move::default(),
        m2: Move::default(),
        m3: Move::default(),
    };

    // Set up moves with PP from movedex
    for (i, move_name) in packed.moves.iter().enumerate() {
        let normalized_move = normalize_name(move_name);
        let move_choice = Choices::from_str(&normalized_move).unwrap_or(Choices::NONE);

        // Look up PP in movedex, default to 32 if not found
        let pp = movedex
            .get(&normalized_move)
            .map(|entry| (entry.pp as f32 * 1.6) as i8)
            .unwrap_or(32);

        let move_data = Move {
            id: move_choice,
            disabled: false,
            pp,
            choice: Default::default(),
        };

        match i {
            0 => moves.m0 = move_data,
            1 => moves.m1 = move_data,
            2 => moves.m2 = move_data,
            3 => moves.m3 = move_data,
            _ => break,
        }
    }

    // Rest of the Pokemon creation remains the same
    Pokemon {
        id: PokemonName::from_str(
            &packed
                .species
                .as_ref()
                .unwrap_or(&packed.nickname)
                .to_uppercase(),
        )
        .unwrap_or(PokemonName::NONE),
        level: 100,
        types: (type1, type2),
        base_types: (type1, type2),
        hp: ((2 * dex_entry.base_stats.hp + 31 + (packed.evs[0] as u16 / 4)) * 100 / 100 + 110)
            as i16,
        maxhp: ((2 * dex_entry.base_stats.hp + 31 + (packed.evs[0] as u16 / 4)) * 100 / 100 + 110)
            as i16,
        ability: Abilities::from_str(&normalize_name(&packed.ability)).unwrap_or(Abilities::NONE),
        base_ability: Abilities::from_str(&normalize_name(&packed.ability))
            .unwrap_or(Abilities::NONE),
        item: Items::from_str(&normalize_name(&packed.item)).unwrap_or(Items::NONE),
        nature: PokemonNature::from_str(&normalize_name(&packed.nature))
            .unwrap_or(PokemonNature::SERIOUS),
        evs: (
            packed.evs[0],
            packed.evs[1],
            packed.evs[2],
            packed.evs[3],
            packed.evs[4],
            packed.evs[5],
        ),
        attack: ((2 * dex_entry.base_stats.attack + 31 + (packed.evs[1] as u16 / 4)) * 100 / 100
            + 5) as i16,
        defense: ((2 * dex_entry.base_stats.defense + 31 + (packed.evs[2] as u16 / 4)) * 100 / 100
            + 5) as i16,
        special_attack: ((2 * dex_entry.base_stats.special_attack
            + 31
            + (packed.evs[3] as u16 / 4))
            * 100
            / 100
            + 5) as i16,
        special_defense: ((2 * dex_entry.base_stats.special_defense
            + 31
            + (packed.evs[4] as u16 / 4))
            * 100
            / 100
            + 5) as i16,
        speed: ((2 * dex_entry.base_stats.speed + 31 + (packed.evs[5] as u16 / 4)) * 100 / 100 + 5)
            as i16,
        status: PokemonStatus::NONE,
        rest_turns: 0,
        sleep_turns: 0,
        weight_kg: dex_entry.weightkg,
        terastallized: false,
        tera_type: PokemonType::from_str(&normalize_name(&packed.tera_type))
            .unwrap_or(PokemonType::NORMAL),
        moves,
    }
}

pub fn initialize_battle_state(
    random_teams_json: &str,
    pokedex_json: &str,
    movedex_json: &str,
) -> State {
    let random_teams: Vec<PackedTeamEntry> =
        serde_json::from_str(random_teams_json).expect("Failed to parse random teams JSON");

    let pokedex: HashMap<String, PokemonDexEntry> =
        serde_json::from_str(pokedex_json).expect("Failed to parse pokedex JSON");

    let movedex: HashMap<String, MoveDexEntry> =
        serde_json::from_str(movedex_json).expect("Failed to parse movedex JSON");

    // Team selection logic remains the same
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let selected_teams: Vec<_> = random_teams.choose_multiple(&mut rng, 2).collect();

    let team1 = parse_packed_team(&selected_teams[0].packed_team);
    let team2 = parse_packed_team(&selected_teams[1].packed_team);

    assert_eq!(team1.len(), 6, "Team 1 doesn't have 6 Pokemon: {:?}", team1);
    assert_eq!(team2.len(), 6, "Team 2 doesn't have 6 Pokemon: {:?}", team2);

    // Create Pokemon with movedex
    let side_one_pokemon = SidePokemon {
        p0: create_pokemon(&team1[0], &pokedex, &movedex),
        p1: create_pokemon(&team1[1], &pokedex, &movedex),
        p2: create_pokemon(&team1[2], &pokedex, &movedex),
        p3: create_pokemon(&team1[3], &pokedex, &movedex),
        p4: create_pokemon(&team1[4], &pokedex, &movedex),
        p5: create_pokemon(&team1[5], &pokedex, &movedex),
    };

    let side_two_pokemon = SidePokemon {
        p0: create_pokemon(&team2[0], &pokedex, &movedex),
        p1: create_pokemon(&team2[1], &pokedex, &movedex),
        p2: create_pokemon(&team2[2], &pokedex, &movedex),
        p3: create_pokemon(&team2[3], &pokedex, &movedex),
        p4: create_pokemon(&team2[4], &pokedex, &movedex),
        p5: create_pokemon(&team2[5], &pokedex, &movedex),
    };

    // Rest of the state initialization remains the same
    let side_one = Side {
        active_index: PokemonIndex::P0,
        baton_passing: false,
        pokemon: side_one_pokemon,
        side_conditions: SideConditions::default(),
        wish: (0, 0),
        future_sight: (0, PokemonIndex::P0),
        force_switch: false,
        force_trapped: false,
        slow_uturn_move: false,
        volatile_statuses: HashSet::new(),
        substitute_health: 0,
        attack_boost: 0,
        defense_boost: 0,
        special_attack_boost: 0,
        special_defense_boost: 0,
        speed_boost: 0,
        accuracy_boost: 0,
        evasion_boost: 0,
        last_used_move: LastUsedMove::None,
        damage_dealt: Default::default(),
        switch_out_move_second_saved_move: Choices::NONE,
    };

    let side_two = Side {
        active_index: PokemonIndex::P0,
        baton_passing: false,
        pokemon: side_two_pokemon,
        side_conditions: SideConditions::default(),
        wish: (0, 0),
        future_sight: (0, PokemonIndex::P0),
        force_switch: false,
        force_trapped: false,
        slow_uturn_move: false,
        volatile_statuses: HashSet::new(),
        substitute_health: 0,
        attack_boost: 0,
        defense_boost: 0,
        special_attack_boost: 0,
        special_defense_boost: 0,
        speed_boost: 0,
        accuracy_boost: 0,
        evasion_boost: 0,
        last_used_move: LastUsedMove::None,
        damage_dealt: Default::default(),
        switch_out_move_second_saved_move: Choices::NONE,
    };

    State {
        side_one,
        side_two,
        weather: StateWeather {
            weather_type: Weather::NONE,
            turns_remaining: 0,
        },
        terrain: StateTerrain {
            terrain_type: Terrain::NONE,
            turns_remaining: 0,
        },
        trick_room: StateTrickRoom {
            active: false,
            turns_remaining: 0,
        },
        team_preview: false,
        use_last_used_move: false,
        use_damage_dealt: false,
    }
}
