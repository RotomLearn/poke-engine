use crate::state::{
    pokemon_index_iter, Pokemon, PokemonStatus, Side, SideReference, State, Weather,
};

pub const HP_BINS: usize = 16;
pub const PP_BINS: usize = 4;

fn get_hp_bin(current_hp: i16, max_hp: i16) -> usize {
    if current_hp == 0 {
        return 0;
    }
    let hp_fraction = (current_hp as f32) / (max_hp as f32);
    let bin = (hp_fraction * HP_BINS as f32).floor() as usize;
    // Ensure we don't exceed the number of bins
    bin.min(HP_BINS - 1) + 1 // Add 1 since 0 is reserved for fainted
}

fn get_pp_bin(pp: i8) -> usize {
    // Using the formula ⌊x^(1/3)⌋ as specified
    (pp as f32).powf(1.0 / 3.0).floor() as usize
}

fn encode_onehot(value: usize, size: usize) -> Vec<f32> {
    let mut vec = vec![0.0; size];
    if value < size {
        vec[value] = 1.0;
    }
    vec
}

fn encode_volatile_statuses(observation: &mut Vec<f32>, side: &Side) {
    // 104 binary values for volatile statuses
    let mut volatile_vec = vec![0.0; 104];
    for status in &side.volatile_statuses {
        let status_idx = *status as usize;
        if status_idx < 104 {
            volatile_vec[status_idx] = 1.0;
        }
    }
    observation.extend(volatile_vec);
}

fn encode_pokemon(pokemon: &Pokemon, side: &Side, is_active: bool) -> Vec<f32> {
    let mut vec = Vec::new();

    // Active flag (2 values)
    vec.extend(encode_onehot(if is_active { 1 } else { 0 }, 2));

    // Species (1 value)
    vec.push(pokemon.id as i32 as f32);

    // Ability (1 value)
    vec.push(pokemon.ability as i32 as f32);

    // Item (1 value)
    vec.push(pokemon.item as i32 as f32);

    // Moves (4 x move_id + PP)
    for m in pokemon.moves.into_iter().take(4) {
        vec.push(m.id as i32 as f32);
        vec.extend(encode_onehot(get_pp_bin(m.pp), PP_BINS));
    }

    // Types (20 binary values for each possible type)
    let mut type_vec = vec![0.0; 20];
    type_vec[pokemon.types.0 as usize] = 1.0;
    if pokemon.types.1 != pokemon.types.0 {
        type_vec[pokemon.types.1 as usize] = 1.0;
    }
    vec.extend(type_vec);

    // Terastallized status (2 values - binary flag)
    vec.extend(encode_onehot(if pokemon.terastallized { 1 } else { 0 }, 2));

    // Tera type (20 binary values for possible tera type)
    let mut tera_type_vec = vec![0.0; 20];
    tera_type_vec[pokemon.tera_type as usize] = 1.0;
    vec.extend(tera_type_vec);

    // HP fraction (HP_BINS + 1 bins)
    vec.extend(encode_onehot(
        get_hp_bin(pokemon.hp, pokemon.maxhp),
        HP_BINS + 1,
    ));

    // Boosts (13 values each, -6 to +6)
    vec.extend(encode_onehot((side.attack_boost + 6) as usize, 13));
    vec.extend(encode_onehot((side.defense_boost + 6) as usize, 13));
    vec.extend(encode_onehot((side.special_attack_boost + 6) as usize, 13));
    vec.extend(encode_onehot((side.special_defense_boost + 6) as usize, 13));
    vec.extend(encode_onehot((side.speed_boost + 6) as usize, 13));
    vec.extend(encode_onehot((side.accuracy_boost + 6) as usize, 13));

    // Status conditions (7 binary values)
    let mut status_vec = vec![0.0; 7];
    match pokemon.status {
        PokemonStatus::NONE => (),
        status => status_vec[status as usize] = 1.0,
    }
    vec.extend(status_vec);

    // Additional counters
    vec.extend(encode_onehot(pokemon.rest_turns as usize, 4)); // Rest turns
    vec.extend(encode_onehot(pokemon.sleep_turns as usize, 4)); // Sleep turns

    vec
}

pub fn generate_observation(state: &State, side_reference: SideReference) -> Vec<f32> {
    let mut observation = Vec::with_capacity(3253);

    // Weather encoding (14 values)
    // First 5 values for weather type (NONE, SUN, RAIN, SAND, HAIL)
    let weather_index = match state.weather.weather_type {
        Weather::NONE => 0,
        Weather::SUN => 1,
        Weather::RAIN => 2,
        Weather::SAND => 3,
        Weather::HAIL => 4,
        _ => 0, // Other weather types treated as NONE
    };
    observation.extend(encode_onehot(weather_index, 5));

    // Next 9 values for turns remaining (0-8)
    if weather_index == 0 {
        observation.extend(encode_onehot(0, 9)); // No turns for no weather
    } else {
        observation.extend(encode_onehot(state.weather.turns_remaining as usize, 9));
    }
    // Trick room (7 values)
    observation.extend(encode_onehot(
        if state.trick_room.active {
            (state.trick_room.turns_remaining as usize).min(6)
        } else {
            0
        },
        7,
    ));

    // Get references to our side and opponent's side
    let (our_side, opponent_side) = state.get_both_sides_immutable(&side_reference);

    // Side conditions for both sides
    // Aurora Veil (9 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.aurora_veil as usize,
        9,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.aurora_veil as usize,
        9,
    ));

    // Crafty Shield (2 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.crafty_shield as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.crafty_shield as usize,
        2,
    ));

    // Healing Wish (2 presence)
    observation.extend(encode_onehot(
        our_side.side_conditions.healing_wish as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.healing_wish as usize,
        2,
    ));

    // Light Screen (9 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.light_screen as usize,
        9,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.light_screen as usize,
        9,
    ));

    // Lucky Chant (6 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.lucky_chant as usize,
        6,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.lucky_chant as usize,
        6,
    ));

    // Lunar Dance (2 presence)
    observation.extend(encode_onehot(
        our_side.side_conditions.lunar_dance as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.lunar_dance as usize,
        2,
    ));

    // Mat Block (2 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.mat_block as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.mat_block as usize,
        2,
    ));

    // Mist (6 turns)
    observation.extend(encode_onehot(our_side.side_conditions.mist as usize, 6));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.mist as usize,
        6,
    ));

    // Protect (2 presence)
    observation.extend(encode_onehot(our_side.side_conditions.protect as usize, 2));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.protect as usize,
        2,
    ));

    // Quick Guard (2 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.quick_guard as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.quick_guard as usize,
        2,
    ));

    // Reflect (9 turns)
    observation.extend(encode_onehot(our_side.side_conditions.reflect as usize, 9));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.reflect as usize,
        9,
    ));

    // Safeguard (7 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.safeguard as usize,
        7,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.safeguard as usize,
        7,
    ));

    // Spikes (4 layers)
    observation.extend(encode_onehot(our_side.side_conditions.spikes as usize, 4));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.spikes as usize,
        4,
    ));

    // Stealth Rock (2 presence)
    observation.extend(encode_onehot(
        our_side.side_conditions.stealth_rock as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.stealth_rock as usize,
        2,
    ));

    // Sticky Web (2 presence)
    observation.extend(encode_onehot(
        our_side.side_conditions.sticky_web as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.sticky_web as usize,
        2,
    ));

    // Tailwind (5 turns)
    observation.extend(encode_onehot(our_side.side_conditions.tailwind as usize, 5));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.tailwind as usize,
        5,
    ));

    // Toxic Spikes (3 layers)
    observation.extend(encode_onehot(
        our_side.side_conditions.toxic_spikes as usize,
        3,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.toxic_spikes as usize,
        3,
    ));

    // Wide Guard (2 turns)
    observation.extend(encode_onehot(
        our_side.side_conditions.wide_guard as usize,
        2,
    ));
    observation.extend(encode_onehot(
        opponent_side.side_conditions.wide_guard as usize,
        2,
    ));

    // Encode volatile statuses for both sides (104 binary values per side)
    encode_volatile_statuses(&mut observation, our_side);
    encode_volatile_statuses(&mut observation, opponent_side);

    // Encode all Pokémon
    // Active Pokémon first
    observation.extend(encode_pokemon(
        our_side.get_active_immutable(),
        our_side,
        true,
    ));
    observation.extend(encode_pokemon(
        opponent_side.get_active_immutable(),
        opponent_side,
        true,
    ));

    // Then rest of the team
    for pokemon_index in pokemon_index_iter() {
        if pokemon_index != our_side.active_index {
            observation.extend(encode_pokemon(
                &our_side.pokemon[pokemon_index],
                our_side,
                false,
            ));
        }
        if pokemon_index != opponent_side.active_index {
            observation.extend(encode_pokemon(
                &opponent_side.pokemon[pokemon_index],
                opponent_side,
                false,
            ));
        }
    }

    observation
}
