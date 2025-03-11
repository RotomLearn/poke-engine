use crate::abilities::Abilities;
use crate::choices::{Choice, Choices, Effect, MoveCategory, MoveTarget, MultiHitMove};
use crate::items::Items;
use crate::state::{PokemonStatus, PokemonType, PokemonVolatileStatus};
use ndarray::Array1;
use std::collections::HashMap;

// Function to create an embedding for a single move
fn create_move_embedding(choice: &Choice) -> Array1<f32> {
    // Define dimensions for each part of the embedding
    const TYPE_DIM: usize = 18; // Number of Pokémon types
    const CATEGORY_DIM: usize = 3; // Physical, Special, Status
    const TARGET_DIM: usize = 2; // Self, Opponent
    const NUMERIC_DIM: usize = 6; // Power, accuracy, priority, etc.
    const FLAGS_DIM: usize = 16; // All the boolean flags
    const EFFECT_DIM: usize = 20; // Status effects, boosts, etc.

    const TOTAL_DIM: usize =
        TYPE_DIM + CATEGORY_DIM + TARGET_DIM + NUMERIC_DIM + FLAGS_DIM + EFFECT_DIM;

    // Initialize the embedding vector with zeros
    let mut embedding = Array1::zeros(TOTAL_DIM);
    let mut idx = 0;

    // Encode move type (one-hot)
    let type_idx = choice.move_type as usize;
    if type_idx < TYPE_DIM {
        embedding[type_idx] = 1.0;
    }
    idx += TYPE_DIM;

    // Encode move category (one-hot)
    match choice.category {
        MoveCategory::Physical => embedding[idx] = 1.0,
        MoveCategory::Special => embedding[idx + 1] = 1.0,
        MoveCategory::Status => embedding[idx + 2] = 1.0,
        _ => {}
    }
    idx += CATEGORY_DIM;

    // Encode target (one-hot)
    match choice.target {
        MoveTarget::User => embedding[idx] = 1.0,
        MoveTarget::Opponent => embedding[idx + 1] = 1.0,
    }
    idx += TARGET_DIM;

    // Encode numeric features (normalized)
    // Base power (normalize to [0, 1] assuming max power is 250)
    embedding[idx] = choice.base_power / 250.0;
    // Accuracy (already in range [0, 1])
    embedding[idx + 1] = choice.accuracy / 100.0;
    // Priority (normalize from [-7, 7] to [0, 1])
    embedding[idx + 2] = (choice.priority as f32 + 7.0) / 14.0;
    // Drain (0 if None)
    embedding[idx + 3] = choice.drain.unwrap_or(0.0);
    // Recoil (0 if None)
    embedding[idx + 4] = choice.recoil.unwrap_or(0.0);
    // Crash (0 if None)
    embedding[idx + 5] = choice.crash.unwrap_or(0.0);
    idx += NUMERIC_DIM;

    // Encode flags (binary)
    embedding[idx] = if choice.flags.bite { 1.0 } else { 0.0 };
    embedding[idx + 1] = if choice.flags.bullet { 1.0 } else { 0.0 };
    embedding[idx + 2] = if choice.flags.charge { 1.0 } else { 0.0 };
    embedding[idx + 3] = if choice.flags.contact { 1.0 } else { 0.0 };
    embedding[idx + 4] = if choice.flags.drag { 1.0 } else { 0.0 };
    embedding[idx + 5] = if choice.flags.heal { 1.0 } else { 0.0 };
    embedding[idx + 6] = if choice.flags.powder { 1.0 } else { 0.0 };
    embedding[idx + 7] = if choice.flags.protect { 1.0 } else { 0.0 };
    embedding[idx + 8] = if choice.flags.pulse { 1.0 } else { 0.0 };
    embedding[idx + 9] = if choice.flags.punch { 1.0 } else { 0.0 };
    embedding[idx + 10] = if choice.flags.recharge { 1.0 } else { 0.0 };
    embedding[idx + 11] = if choice.flags.reflectable { 1.0 } else { 0.0 };
    embedding[idx + 12] = if choice.flags.slicing { 1.0 } else { 0.0 };
    embedding[idx + 13] = if choice.flags.sound { 1.0 } else { 0.0 };
    embedding[idx + 14] = if choice.flags.pivot { 1.0 } else { 0.0 };
    embedding[idx + 15] = if choice.flags.wind { 1.0 } else { 0.0 };
    idx += FLAGS_DIM;

    // Encode special properties
    embedding[idx] = if choice.multi_hit() != MultiHitMove::None {
        1.0
    } else {
        0.0
    };
    embedding[idx + 1] = if choice.move_id.increased_crit_ratio() {
        1.0
    } else {
        0.0
    };
    embedding[idx + 2] = if choice.move_id.guaranteed_crit() {
        1.0
    } else {
        0.0
    };

    // Encode status effects
    if let Some(status) = &choice.status {
        match status.status {
            PokemonStatus::BURN => embedding[idx + 3] = 1.0,
            PokemonStatus::FREEZE => embedding[idx + 4] = 1.0,
            PokemonStatus::PARALYZE => embedding[idx + 5] = 1.0,
            PokemonStatus::POISON => embedding[idx + 6] = 1.0,
            PokemonStatus::SLEEP => embedding[idx + 7] = 1.0,
            PokemonStatus::TOXIC => embedding[idx + 8] = 1.0,
            _ => {}
        }
    }

    // Encode boosting effects
    if let Some(boost) = &choice.boost {
        let boost_idx = idx + 9;
        let boosts = &boost.boosts;

        // Add a normalized value for each stat boost [-6, 6] to [0, 1]
        embedding[boost_idx] = (boosts.attack as f32 + 6.0) / 12.0;
        embedding[boost_idx + 1] = (boosts.defense as f32 + 6.0) / 12.0;
        embedding[boost_idx + 2] = (boosts.special_attack as f32 + 6.0) / 12.0;
        embedding[boost_idx + 3] = (boosts.special_defense as f32 + 6.0) / 12.0;
        embedding[boost_idx + 4] = (boosts.speed as f32 + 6.0) / 12.0;
        embedding[boost_idx + 5] = (boosts.accuracy as f32 + 6.0) / 12.0;

        // Direction of boost (self vs opponent)
        embedding[boost_idx + 6] = match boost.target {
            MoveTarget::User => 1.0,
            MoveTarget::Opponent => 0.0,
        };
    }

    // Encode secondary effects (cumulative probability-weighted effects)
    if let Some(secondaries) = &choice.secondaries {
        for secondary in secondaries {
            let chance = secondary.chance / 100.0;
            match &secondary.effect {
                Effect::Status(status) => match status {
                    PokemonStatus::BURN => embedding[idx + 3] += chance,
                    PokemonStatus::FREEZE => embedding[idx + 4] += chance,
                    PokemonStatus::PARALYZE => embedding[idx + 5] += chance,
                    PokemonStatus::POISON => embedding[idx + 6] += chance,
                    PokemonStatus::SLEEP => embedding[idx + 7] += chance,
                    PokemonStatus::TOXIC => embedding[idx + 8] += chance,
                    _ => {}
                },
                Effect::VolatileStatus(vol_status) => {
                    // Handle volatile statuses like confusion, flinch, etc.
                    if *vol_status == PokemonVolatileStatus::CONFUSION {
                        embedding[idx + 16] += chance;
                    } else if *vol_status == PokemonVolatileStatus::FLINCH {
                        embedding[idx + 17] += chance;
                    }
                }
                Effect::Boost(boosts) => {
                    let boost_idx = idx + 9;

                    // Add probability-weighted boost values
                    // Convert from [-6,6] to [0,1] and multiply by chance
                    let normalized_attack = (boosts.attack as f32 + 6.0) / 12.0 * chance;
                    let normalized_defense = (boosts.defense as f32 + 6.0) / 12.0 * chance;
                    let normalized_special_attack =
                        (boosts.special_attack as f32 + 6.0) / 12.0 * chance;
                    let normalized_special_defense =
                        (boosts.special_defense as f32 + 6.0) / 12.0 * chance;
                    let normalized_speed = (boosts.speed as f32 + 6.0) / 12.0 * chance;
                    let normalized_accuracy = (boosts.accuracy as f32 + 6.0) / 12.0 * chance;

                    // Add to the embedding based on the target
                    if secondary.target == MoveTarget::User {
                        // User-targeted boosts (positive for user means higher values)
                        embedding[boost_idx] += normalized_attack;
                        embedding[boost_idx + 1] += normalized_defense;
                        embedding[boost_idx + 2] += normalized_special_attack;
                        embedding[boost_idx + 3] += normalized_special_defense;
                        embedding[boost_idx + 4] += normalized_speed;
                        embedding[boost_idx + 5] += normalized_accuracy;
                    } else {
                        // Opponent-targeted boosts (negative for opponent means higher values for those dimensions)
                        // For opponent, we invert the boost value since lowering opponent stats is beneficial
                        embedding[boost_idx] -= normalized_attack;
                        embedding[boost_idx + 1] -= normalized_defense;
                        embedding[boost_idx + 2] -= normalized_special_attack;
                        embedding[boost_idx + 3] -= normalized_special_defense;
                        embedding[boost_idx + 4] -= normalized_speed;
                        embedding[boost_idx + 5] -= normalized_accuracy;
                    }

                    // Update the target indicator with the probability
                    if secondary.target == MoveTarget::User {
                        embedding[boost_idx + 6] += chance;
                    } else {
                        embedding[boost_idx + 6] -= chance;
                    }
                }
                Effect::Heal(amount) => {
                    // Add healing effects at position idx + 18
                    embedding[idx + 18] += *amount * chance;
                }
                Effect::RemoveItem => {
                    // Add item removal effect at position idx + 19
                    embedding[idx + 19] += chance;
                }
            }
        }
    }
    embedding
}

pub fn create_all_move_embeddings(
    moves: &HashMap<Choices, Choice>,
) -> HashMap<Choices, Array1<f32>> {
    let mut embeddings = HashMap::new();

    for (choice_id, choice) in moves.iter() {
        let embedding = create_move_embedding(choice);
        embeddings.insert(*choice_id, embedding);
    }

    embeddings
}

fn create_item_embedding(item: Items) -> Array1<f32> {
    // Define dimensions for each part of the embedding
    const TYPE_DIM: usize = 18; // Number of Pokémon types
    const EFFECT_CATEGORY_DIM: usize = 8; // Boost stats, heal, reduce damage, etc.
    const TARGET_DIM: usize = 2; // Self, Opponent
    const TIMING_DIM: usize = 4; // Before move, on switch, end of turn, during attack
    const CONSUMABLE_DIM: usize = 1; // Whether the item is consumed on use

    const TOTAL_DIM: usize =
        TYPE_DIM + EFFECT_CATEGORY_DIM + TARGET_DIM + TIMING_DIM + CONSUMABLE_DIM;

    // Initialize the embedding vector with zeros
    let mut embedding = Array1::zeros(TOTAL_DIM);
    let mut idx = 0;

    // Encode type-specific effects (one-hot for affected types)
    match item {
        // Type-boosting items
        Items::BLACKBELT => embedding[PokemonType::FIGHTING as usize] = 1.0,
        Items::BLACKGLASSES => embedding[PokemonType::DARK as usize] = 1.0,
        Items::CHARCOAL => embedding[PokemonType::FIRE as usize] = 1.0,
        Items::DRAGONFANG | Items::DRAGONSCALE => embedding[PokemonType::DRAGON as usize] = 1.0,
        Items::FAIRYFEATHER => embedding[PokemonType::FAIRY as usize] = 1.0,
        Items::METALCOAT => embedding[PokemonType::STEEL as usize] = 1.0,
        Items::MYSTICWATER | Items::SEAINCENSE | Items::WAVEINCENSE => {
            embedding[PokemonType::WATER as usize] = 1.0
        }
        Items::NEVERMELTICE => embedding[PokemonType::ICE as usize] = 1.0,
        Items::PINKBOW | Items::POLKADOTBOW | Items::SILKSCARF => {
            embedding[PokemonType::NORMAL as usize] = 1.0
        }
        Items::POISONBARB => embedding[PokemonType::POISON as usize] = 1.0,
        Items::SHARPBEAK => embedding[PokemonType::FLYING as usize] = 1.0,
        Items::SILVERPOWDER => embedding[PokemonType::BUG as usize] = 1.0,
        Items::SOFTSAND => embedding[PokemonType::GROUND as usize] = 1.0,
        Items::SPELLTAG => embedding[PokemonType::GHOST as usize] = 1.0,
        Items::MIRACLESEED => embedding[PokemonType::GRASS as usize] = 1.0,
        Items::TWISTEDSPOON | Items::ODDINCENSE => embedding[PokemonType::PSYCHIC as usize] = 1.0,
        Items::HARDSTONE => embedding[PokemonType::ROCK as usize] = 1.0,
        Items::MAGNET => embedding[PokemonType::ELECTRIC as usize] = 1.0,

        // Type-specific gems
        Items::NORMALGEM => embedding[PokemonType::NORMAL as usize] = 1.0,
        Items::BUGGEM => embedding[PokemonType::BUG as usize] = 1.0,
        Items::ELECTRICGEM => embedding[PokemonType::ELECTRIC as usize] = 1.0,
        Items::FIGHTINGGEM => embedding[PokemonType::FIGHTING as usize] = 1.0,
        Items::GHOSTGEM => embedding[PokemonType::GHOST as usize] = 1.0,
        Items::PSYCHICGEM => embedding[PokemonType::PSYCHIC as usize] = 1.0,
        Items::FLYINGGEM => embedding[PokemonType::FLYING as usize] = 1.0,
        Items::STEELGEM => embedding[PokemonType::STEEL as usize] = 1.0,
        Items::ICEGEM => embedding[PokemonType::ICE as usize] = 1.0,
        Items::POISONGEM => embedding[PokemonType::POISON as usize] = 1.0,
        Items::FIREGEM => embedding[PokemonType::FIRE as usize] = 1.0,
        Items::DRAGONGEM => embedding[PokemonType::DRAGON as usize] = 1.0,
        Items::GROUNDGEM => embedding[PokemonType::GROUND as usize] = 1.0,
        Items::WATERGEM => embedding[PokemonType::WATER as usize] = 1.0,
        Items::DARKGEM => embedding[PokemonType::DARK as usize] = 1.0,
        Items::ROCKGEM => embedding[PokemonType::ROCK as usize] = 1.0,
        Items::GRASSGEM => embedding[PokemonType::GRASS as usize] = 1.0,
        Items::FAIRYGEM => embedding[PokemonType::FAIRY as usize] = 1.0,

        // Plates that affect both type and power
        Items::FISTPLATE => embedding[PokemonType::FIGHTING as usize] = 1.0,
        Items::SKYPLATE => embedding[PokemonType::FLYING as usize] = 1.0,
        Items::TOXICPLATE => embedding[PokemonType::POISON as usize] = 1.0,
        Items::EARTHPLATE => embedding[PokemonType::GROUND as usize] = 1.0,
        Items::STONEPLATE => embedding[PokemonType::ROCK as usize] = 1.0,
        Items::INSECTPLATE => embedding[PokemonType::BUG as usize] = 1.0,
        Items::SPOOKYPLATE => embedding[PokemonType::GHOST as usize] = 1.0,
        Items::IRONPLATE => embedding[PokemonType::STEEL as usize] = 1.0,
        Items::FLAMEPLATE => embedding[PokemonType::FIRE as usize] = 1.0,
        Items::SPLASHPLATE => embedding[PokemonType::WATER as usize] = 1.0,
        Items::MEADOWPLATE => embedding[PokemonType::GRASS as usize] = 1.0,
        Items::ZAPPLATE => embedding[PokemonType::ELECTRIC as usize] = 1.0,
        Items::MINDPLATE => embedding[PokemonType::PSYCHIC as usize] = 1.0,
        Items::ICICLEPLATE => embedding[PokemonType::ICE as usize] = 1.0,
        Items::DRACOPLATE => embedding[PokemonType::DRAGON as usize] = 1.0,
        Items::DREADPLATE => embedding[PokemonType::DARK as usize] = 1.0,
        Items::PIXIEPLATE => embedding[PokemonType::FAIRY as usize] = 1.0,

        // Type-specific damage reduction berries
        Items::CHOPLEBERRY => embedding[PokemonType::FIGHTING as usize] = -1.0,
        Items::BABIRIBERRY => embedding[PokemonType::STEEL as usize] = -1.0,
        Items::CHARTIBERRY => embedding[PokemonType::ROCK as usize] = -1.0,
        Items::CHILANBERRY => embedding[PokemonType::NORMAL as usize] = -1.0,
        Items::COBABERRY => embedding[PokemonType::FLYING as usize] = -1.0,
        Items::COLBURBERRY => embedding[PokemonType::DARK as usize] = -1.0,
        Items::HABANBERRY => embedding[PokemonType::DRAGON as usize] = -1.0,
        Items::KASIBBERRY => embedding[PokemonType::GHOST as usize] = -1.0,
        Items::KEBIABERRY => embedding[PokemonType::POISON as usize] = -1.0,
        Items::OCCABERRY => embedding[PokemonType::FIRE as usize] = -1.0,
        Items::PASSHOBERRY => embedding[PokemonType::WATER as usize] = -1.0,
        Items::PAYAPABERRY => embedding[PokemonType::PSYCHIC as usize] = -1.0,
        Items::RINDOBERRY => embedding[PokemonType::GRASS as usize] = -1.0,
        Items::ROSELIBERRY => embedding[PokemonType::FAIRY as usize] = -1.0,
        Items::SHUCABERRY => embedding[PokemonType::GROUND as usize] = -1.0,
        Items::TANGABERRY => embedding[PokemonType::BUG as usize] = -1.0,
        Items::WACANBERRY => embedding[PokemonType::ELECTRIC as usize] = -1.0,
        Items::YACHEBERRY => embedding[PokemonType::ICE as usize] = -1.0,

        _ => {}
    }
    idx += TYPE_DIM;

    // Encode effect categories
    match item {
        // Boost attack
        Items::CHOICEBAND | Items::THICKCLUB | Items::LIECHIBERRY | Items::CELLBATTERY => {
            embedding[idx] = 1.0;
        }
        // Boost special attack
        Items::CHOICESPECS
        | Items::WISEGLASSES
        | Items::PETAYABERRY
        | Items::ABSORBBULB
        | Items::THROATSPRAY => {
            embedding[idx + 1] = 1.0;
        }
        // Boost speed
        Items::CHOICESCARF | Items::SALACBERRY => {
            embedding[idx + 2] = 1.0;
        }
        // Boost defense
        Items::EVIOLITE | Items::METALPOWDER | Items::ELECTRICSEED | Items::GRASSYSEED => {
            embedding[idx + 3] = 1.0;
        }
        // Boost special defense
        Items::ASSAULTVEST | Items::MISTYSEED | Items::PSYCHICSEED => {
            embedding[idx + 4] = 1.0;
        }
        // Healing effects
        Items::LEFTOVERS | Items::BLACKSLUDGE | Items::SHELLBELL | Items::SITRUSBERRY => {
            embedding[idx + 5] = 1.0;
        }
        // Status healing
        Items::LUMBERRY => {
            embedding[idx + 6] = 1.0;
        }
        // Damage modification (Life Orb, Expert Belt)
        Items::LIFEORB | Items::EXPERTBELT | Items::WEAKNESSPOLICY => {
            embedding[idx + 7] = 1.0;
        }
        _ => {}
    }
    idx += EFFECT_CATEGORY_DIM;

    // Encode target (self vs opponent)
    match item {
        // Self-affecting items
        Items::CHOICEBAND
        | Items::CHOICESPECS
        | Items::CHOICESCARF
        | Items::LEFTOVERS
        | Items::LIFEORB
        | Items::SHELLBELL
        | Items::LUMBERRY
        | Items::SITRUSBERRY
        | Items::PETAYABERRY
        | Items::SALACBERRY
        | Items::LIECHIBERRY => {
            embedding[idx] = 1.0;
        }
        // Opponent-affecting items
        Items::ROCKYHELMET => {
            embedding[idx + 1] = 1.0;
        }
        // Both or depends on context
        Items::WEAKNESSPOLICY => {
            embedding[idx] = 0.5;
            embedding[idx + 1] = 0.5;
        }
        _ => {}
    }
    idx += TARGET_DIM;

    // Encode timing of effect
    match item {
        // Before move
        Items::POWERHERB | Items::CHOICESPECS | Items::CHOICEBAND | Items::CHOICESCARF => {
            embedding[idx] = 1.0;
        }
        // On switch in
        Items::ELECTRICSEED | Items::GRASSYSEED | Items::MISTYSEED | Items::PSYCHICSEED => {
            embedding[idx + 1] = 1.0;
        }
        // End of turn
        Items::LEFTOVERS | Items::BLACKSLUDGE | Items::FLAMEORB | Items::TOXICORB => {
            embedding[idx + 2] = 1.0;
        }
        // During attack
        Items::LIFEORB
        | Items::EXPERTBELT
        | Items::WISEGLASSES
        | Items::MUSCLEBAND
        | Items::CHOICEBAND
        | Items::CHOICESPECS => {
            embedding[idx + 3] = 1.0;
        }
        _ => {}
    }
    idx += TIMING_DIM;

    // Encode consumable property
    match item {
        // Consumable items
        Items::LUMBERRY
        | Items::SITRUSBERRY
        | Items::PETAYABERRY
        | Items::SALACBERRY
        | Items::LIECHIBERRY
        | Items::CUSTAPBERRY
        | Items::NORMALGEM
        | Items::BUGGEM
        | Items::ELECTRICGEM
        | Items::FIGHTINGGEM
        | Items::GHOSTGEM
        | Items::PSYCHICGEM
        | Items::FLYINGGEM
        | Items::STEELGEM
        | Items::ICEGEM
        | Items::POISONGEM
        | Items::FIREGEM
        | Items::DRAGONGEM
        | Items::GROUNDGEM
        | Items::WATERGEM
        | Items::DARKGEM
        | Items::ROCKGEM
        | Items::GRASSGEM
        | Items::FAIRYGEM
        | Items::CHOPLEBERRY
        | Items::BABIRIBERRY
        | Items::CHARTIBERRY
        | Items::CHILANBERRY
        | Items::COBABERRY
        | Items::COLBURBERRY
        | Items::HABANBERRY
        | Items::KASIBBERRY
        | Items::KEBIABERRY
        | Items::OCCABERRY
        | Items::PASSHOBERRY
        | Items::PAYAPABERRY
        | Items::RINDOBERRY
        | Items::ROSELIBERRY
        | Items::SHUCABERRY
        | Items::TANGABERRY
        | Items::WACANBERRY
        | Items::YACHEBERRY
        | Items::ELECTRICSEED
        | Items::GRASSYSEED
        | Items::MISTYSEED
        | Items::PSYCHICSEED
        | Items::THROATSPRAY
        | Items::WEAKNESSPOLICY
        | Items::CELLBATTERY
        | Items::ABSORBBULB => {
            embedding[idx] = 1.0;
        }
        _ => {}
    }

    embedding
}

/// Function to create an embedding for a single ability
pub fn create_ability_embedding(ability: &Abilities) -> Array1<f32> {
    // Define dimensions for each part of the embedding
    const TYPE_DIM: usize = 18; // Number of Pokémon types affected
    const STAT_BOOST_DIM: usize = 6; // Attack, Defense, SpAtk, SpDef, Speed, Accuracy
    const STAT_REDUCE_DIM: usize = 6; // Attack, Defense, SpAtk, SpDef, Speed, Accuracy reductions
    const TERRAIN_WEATHER_DIM: usize = 9; // Weather and terrain effects
    const STATUS_EFFECTS_DIM: usize = 10; // Status effects and immunities
    const DAMAGE_MODIFY_DIM: usize = 4; // Damage modification properties
    const TRIGGER_CONDITIONS_DIM: usize = 8; // When ability activates
    const MOVE_EFFECTS_DIM: usize = 10; // Effects on moves
    const MISC_EFFECTS_DIM: usize = 10; // Other miscellaneous effects

    const TOTAL_DIM: usize = TYPE_DIM
        + STAT_BOOST_DIM
        + STAT_REDUCE_DIM
        + TERRAIN_WEATHER_DIM
        + STATUS_EFFECTS_DIM
        + DAMAGE_MODIFY_DIM
        + TRIGGER_CONDITIONS_DIM
        + MOVE_EFFECTS_DIM
        + MISC_EFFECTS_DIM;

    // Initialize the embedding vector with zeros
    let mut embedding = Array1::zeros(TOTAL_DIM);
    let mut idx = 0;

    // Encode type-specific effects or interactions
    match ability {
        // Type boosting abilities
        Abilities::DRAGONSMAW => embedding[PokemonType::DRAGON as usize] = 1.0,
        Abilities::STEELWORKER | Abilities::STEELYSPIRIT => {
            embedding[PokemonType::STEEL as usize] = 1.0
        }
        Abilities::TRANSISTOR => embedding[PokemonType::ELECTRIC as usize] = 1.0,
        Abilities::BLAZE => embedding[PokemonType::FIRE as usize] = 1.0,
        Abilities::OVERGROW => embedding[PokemonType::GRASS as usize] = 1.0,
        Abilities::TORRENT => embedding[PokemonType::WATER as usize] = 1.0,
        Abilities::SWARM => embedding[PokemonType::BUG as usize] = 1.0,
        Abilities::WATERBUBBLE => embedding[PokemonType::WATER as usize] = 1.0,
        Abilities::DARKAURA => embedding[PokemonType::DARK as usize] = 1.0,
        Abilities::FAIRYAURA => embedding[PokemonType::FAIRY as usize] = 1.0,
        Abilities::ROCKYPAYLOAD => embedding[PokemonType::ROCK as usize] = 1.0,
        Abilities::AERILATE => embedding[PokemonType::FLYING as usize] = 1.0,
        Abilities::REFRIGERATE => embedding[PokemonType::ICE as usize] = 1.0,
        Abilities::PIXILATE => embedding[PokemonType::FAIRY as usize] = 1.0,
        Abilities::GALVANIZE => embedding[PokemonType::ELECTRIC as usize] = 1.0,

        // Type immunity/resistance/weakness abilities
        Abilities::THICKFAT => {
            embedding[PokemonType::FIRE as usize] = -0.5;
            embedding[PokemonType::ICE as usize] = -0.5;
        }
        Abilities::LEVITATE => embedding[PokemonType::GROUND as usize] = -1.0,
        Abilities::WATERABSORB | Abilities::STORMDRAIN => {
            embedding[PokemonType::WATER as usize] = -1.0
        }
        Abilities::VOLTABSORB | Abilities::LIGHTNINGROD | Abilities::MOTORDRIVE => {
            embedding[PokemonType::ELECTRIC as usize] = -1.0
        }
        Abilities::FLASHFIRE => embedding[PokemonType::FIRE as usize] = -1.0,
        Abilities::EARTHEATER => embedding[PokemonType::GROUND as usize] = -1.0,
        Abilities::DRYSKIN => {
            embedding[PokemonType::WATER as usize] = -1.0;
            embedding[PokemonType::FIRE as usize] = 0.25; // Takes more damage from Fire
        }
        Abilities::FLUFFY => {
            embedding[PokemonType::FIRE as usize] = 1.0; // Takes more damage from Fire
        }
        Abilities::WONDERGUARD => {
            // Immune to all non-super-effective damage
            for i in 0..TYPE_DIM {
                embedding[i] = -0.5;
            }
        }
        Abilities::SAPSIPPER => embedding[PokemonType::GRASS as usize] = -1.0,
        Abilities::PURIFYINGSALT => embedding[PokemonType::GHOST as usize] = -0.5,
        Abilities::JUSTIFIED => embedding[PokemonType::DARK as usize] = -0.5, // Not immunity but benefits when hit
        Abilities::SOUNDPROOF => embedding[PokemonType::NORMAL as usize] = -0.5, // Not type but sound moves
        Abilities::BULLETPROOF => embedding[PokemonType::NORMAL as usize] = -0.5, // Not type but bullet moves
        Abilities::STEAMENGINE => {
            embedding[PokemonType::WATER as usize] = -0.5;
            embedding[PokemonType::FIRE as usize] = -0.5;
        }
        Abilities::WELLBAKEDBODY => embedding[PokemonType::FIRE as usize] = -1.0,

        _ => {}
    }
    idx += TYPE_DIM;

    // Encode stat boost effects
    match ability {
        // Attack boosting
        Abilities::INTREPIDSWORD
        | Abilities::SUPREMEOVERLORD
        | Abilities::HUGEPOWER
        | Abilities::PUREPOWER
        | Abilities::EMBODYASPECTHEARTHFLAME => embedding[idx] = 1.0,

        // Defense boosting
        Abilities::DAUNTLESSSHIELD
        | Abilities::EMBODYASPECTCORNERSTONE
        | Abilities::GRASSPELT
        | Abilities::FURCOAT
        | Abilities::MARVELSCALE
        | Abilities::SHIELDSDOWN => embedding[idx + 1] = 1.0,

        // Special Attack boosting
        Abilities::SOLARPOWER
        | Abilities::HADRONENGINE
        | Abilities::CHILLINGNEIGH
        | Abilities::GRIMNEIGH
        | Abilities::ASONESPECTRIER => embedding[idx + 2] = 1.0,

        // Special Defense boosting
        Abilities::EMBODYASPECTWELLSPRING
        | Abilities::ICESCALES
        | Abilities::MULTISCALE
        | Abilities::SHADOWSHIELD => embedding[idx + 3] = 1.0,

        // Speed boosting
        Abilities::EMBODYASPECTTEAL
        | Abilities::SPEEDBOOST
        | Abilities::SWIFTSWIM
        | Abilities::SLUSHRUSH
        | Abilities::CHLOROPHYLL
        | Abilities::SANDFORCE
        | Abilities::SANDRUSH
        | Abilities::QUICKFEET => embedding[idx + 4] = 1.0,

        // Accuracy boosting
        Abilities::COMPOUNDEYES | Abilities::VICTORYSTAR | Abilities::NOGUARD => {
            embedding[idx + 5] = 1.0
        }

        // Multiple stats
        Abilities::DOWNLOAD => {
            embedding[idx] = 0.5; // Attack
            embedding[idx + 2] = 0.5; // Special Attack
        }
        Abilities::BEASTBOOST => {
            // Can boost any stat that's highest
            embedding[idx] = 0.2;
            embedding[idx + 1] = 0.2;
            embedding[idx + 2] = 0.2;
            embedding[idx + 3] = 0.2;
            embedding[idx + 4] = 0.2;
        }
        Abilities::GORILLATACTICS => {
            embedding[idx] = 1.0; // Attack boost
            embedding[idx + 4] = -0.5; // But locks user into one move
        }
        Abilities::BATTLEBOND => {
            embedding[idx] = 0.33;
            embedding[idx + 2] = 0.33;
            embedding[idx + 4] = 0.33;
        }
        Abilities::STEAMENGINE => embedding[idx + 4] = 2.0, // Large speed boost

        _ => {}
    }
    idx += STAT_BOOST_DIM;

    // Encode stat reduction effects
    match ability {
        // Reduces opponent's Attack
        Abilities::INTIMIDATE => embedding[idx] = 1.0,

        // Reduces opponent's Defense
        Abilities::WEAKARMOR => embedding[idx + 1] = 1.0,

        // Reduces opponent's Special Attack
        // (No common abilities directly reduce SpAtk)

        // Reduces opponent's Special Defense
        Abilities::BEADSOFRUIN => embedding[idx + 3] = 1.0,

        // Reduces opponent's Speed
        Abilities::GOOEY | Abilities::TANGLINGHAIR | Abilities::COTTONDOWN => {
            embedding[idx + 4] = 1.0
        }

        // Reduces opponent's Accuracy
        Abilities::SANDVEIL | Abilities::SNOWCLOAK | Abilities::TANGLEDFEET => {
            embedding[idx + 5] = 1.0
        }

        // Multiple stat reductions
        Abilities::TABLETSOFRUIN => embedding[idx] = 1.0, // Reduces opponent's Attack
        Abilities::VESSELOFRUIN => embedding[idx + 2] = 1.0, // Reduces opponent's Special Attack

        _ => {}
    }
    idx += STAT_REDUCE_DIM;

    // Encode terrain and weather effects
    match ability {
        // Weather setting
        Abilities::DROUGHT | Abilities::ORICHALCUMPULSE => embedding[idx] = 1.0, // Sun
        Abilities::SANDSTREAM => embedding[idx + 1] = 1.0,                       // Sand
        Abilities::SNOWWARNING => embedding[idx + 2] = 1.0,                      // Snow/Hail
        Abilities::DRIZZLE => embedding[idx + 3] = 1.0,                          // Rain
        Abilities::DESOLATELAND => embedding[idx] = 2.0, // Harsh Sun (stronger)
        Abilities::PRIMORDIALSEA => embedding[idx + 3] = 2.0, // Heavy Rain (stronger)

        // Terrain setting
        Abilities::ELECTRICSURGE | Abilities::HADRONENGINE => embedding[idx + 4] = 1.0, // Electric Terrain
        Abilities::PSYCHICSURGE => embedding[idx + 5] = 1.0, // Psychic Terrain
        Abilities::MISTYSURGE => embedding[idx + 6] = 1.0,   // Misty Terrain
        Abilities::GRASSYSURGE => embedding[idx + 7] = 1.0,  // Grassy Terrain

        // Weather immunity
        Abilities::AIRLOCK | Abilities::CLOUDNINE => embedding[idx + 8] = 1.0,

        // Weather interaction
        Abilities::SWIFTSWIM => embedding[idx + 3] = 0.5, // Benefits from Rain
        Abilities::SLUSHRUSH => embedding[idx + 2] = 0.5, // Benefits from Hail/Snow
        Abilities::CHLOROPHYLL => embedding[idx] = 0.5,   // Benefits from Sun
        Abilities::FORECAST => {
            embedding[idx] = 0.33; // Sun
            embedding[idx + 2] = 0.33; // Hail/Snow
            embedding[idx + 3] = 0.33; // Rain
        }
        Abilities::SANDFORCE | Abilities::SANDRUSH => embedding[idx + 1] = 0.5, // Benefits from Sand
        Abilities::RAINDISH => embedding[idx + 3] = 0.5, // Benefits from Rain
        Abilities::SOLARPOWER => embedding[idx] = 0.5,   // Benefits from Sun but takes damage
        Abilities::ICEBODY => embedding[idx + 2] = 0.5,  // Benefits from Hail
        Abilities::PROTOSYNTHESIS => embedding[idx] = 0.5, // Activates in Sun
        Abilities::QUARKDRIVE => embedding[idx + 4] = 0.5, // Activates in Electric Terrain

        _ => {}
    }
    idx += TERRAIN_WEATHER_DIM;

    // Encode status effect interactions
    match ability {
        // Status immunity
        Abilities::IMMUNITY => embedding[idx] = 1.0, // Poison immunity
        Abilities::LIMBER => embedding[idx + 1] = 1.0, // Paralysis immunity
        Abilities::WATERVEIL | Abilities::WATERBUBBLE => embedding[idx + 2] = 1.0, // Burn immunity
        Abilities::INSOMNIA | Abilities::VITALSPIRIT => embedding[idx + 3] = 1.0, // Sleep immunity
        Abilities::MAGMAARMOR => embedding[idx + 4] = 1.0, // Freeze immunity
        Abilities::COMATOSE => {
            // Acts as if asleep but immune to status
            for i in 0..5 {
                embedding[idx + i] = 1.0;
            }
        }
        Abilities::SHIELDDUST => embedding[idx + 5] = 1.0, // Immune to secondary effects
        Abilities::LEAFGUARD => {
            // Status immunity in Sun
            for i in 0..5 {
                embedding[idx + i] = 0.5; // Partial value as it's conditional
            }
        }

        // Status healing
        Abilities::NATURALCURE => embedding[idx + 6] = 1.0, // Heals status on switch
        Abilities::SHEDSKIN => embedding[idx + 6] = 0.33,   // 33% chance to heal status
        Abilities::HYDRATION => embedding[idx + 6] = 0.5,   // Heals status in Rain

        // Status causing
        Abilities::POISONTOUCH => embedding[idx] = -0.5, // Can cause poison
        Abilities::TOXICCHAIN => embedding[idx] = -1.0,  // Can cause toxic (worse than poison)
        Abilities::FLAMEBODY => embedding[idx + 2] = -0.5, // Can cause burn
        Abilities::STATIC => embedding[idx + 1] = -0.5,  // Can cause paralysis
        Abilities::EFFECTSPORE => {
            embedding[idx] = -0.33; // Poison
            embedding[idx + 1] = -0.33; // Paralysis
            embedding[idx + 3] = -0.33; // Sleep
        }
        Abilities::POISONPOINT => embedding[idx] = -0.5, // Can cause poison

        // Status benefit
        Abilities::GUTS => embedding[idx + 7] = 1.0, // Boosts Attack when statused
        Abilities::QUICKFEET => embedding[idx + 8] = 1.0, // Boosts Speed when statused
        Abilities::MARVELSCALE => embedding[idx + 7] = 0.5, // Boosts Defense when statused
        Abilities::POISONHEAL => embedding[idx + 9] = 1.0, // Heals from poison
        Abilities::TOXICBOOST => embedding[idx + 7] = 0.5, // Boosts from Toxic status

        _ => {}
    }
    idx += STATUS_EFFECTS_DIM;

    // Encode damage modification properties
    match ability {
        // Damage reduction
        Abilities::MULTISCALE | Abilities::SHADOWSHIELD => embedding[idx] = 0.5, // Reduces damage at full HP
        Abilities::SOLIDROCK | Abilities::FILTER | Abilities::PRISMARMOR => {
            embedding[idx + 1] = 0.5
        } // Reduces super-effective damage
        Abilities::THICKFAT => embedding[idx + 2] = 0.5, // Reduces specific type damage
        Abilities::ICESCALES => embedding[idx + 3] = 0.5, // Reduces special attack damage
        Abilities::FURCOAT => embedding[idx + 3] = 0.5,  // Reduces physical attack damage

        // Damage increase
        Abilities::HUGEPOWER | Abilities::PUREPOWER => embedding[idx] = -1.0, // Doubles Attack
        Abilities::WATERBUBBLE => embedding[idx] = -0.5,                      // Boosts Water moves
        Abilities::TRANSISTOR => embedding[idx] = -0.5, // Boosts Electric moves
        Abilities::DRAGONSMAW => embedding[idx] = -0.5, // Boosts Dragon moves
        Abilities::STEELWORKER => embedding[idx] = -0.5, // Boosts Steel moves
        Abilities::TOUGHCLAWS => embedding[idx] = -0.3, // Boosts contact moves
        Abilities::TECHNICIAN => embedding[idx] = -0.3, // Boosts weak moves
        Abilities::STRONGJAW => embedding[idx] = -0.3,  // Boosts bite moves
        Abilities::IRONFIST => embedding[idx] = -0.3,   // Boosts punch moves
        Abilities::SHARPNESS => embedding[idx] = -0.3,  // Boosts slicing moves
        Abilities::ANALYTIC => embedding[idx] = -0.3,   // Boosts moves when moving last
        Abilities::ROCKYPAYLOAD => embedding[idx] = -0.5, // Boosts Rock moves
        Abilities::PUNKROCK => embedding[idx] = -0.3,   // Boosts sound moves
        Abilities::AERILATE
        | Abilities::GALVANIZE
        | Abilities::PIXILATE
        | Abilities::REFRIGERATE => embedding[idx] = -0.5, // Boosts Normal-type conversions

        // Recoil effects
        Abilities::ROCKHEAD => embedding[idx + 2] = 0.5, // Prevents recoil damage

        _ => {}
    }
    idx += DAMAGE_MODIFY_DIM;

    // Encode trigger conditions
    match ability {
        // On switch-in
        Abilities::INTIMIDATE
        | Abilities::DOWNLOAD
        | Abilities::SCREENCLEANER
        | Abilities::ELECTRICSURGE
        | Abilities::PSYCHICSURGE
        | Abilities::MISTYSURGE
        | Abilities::GRASSYSURGE
        | Abilities::SANDSTREAM
        | Abilities::DROUGHT
        | Abilities::DRIZZLE
        | Abilities::SNOWWARNING
        | Abilities::TRACE => embedding[idx] = 1.0,

        // On hit/contact
        Abilities::ROUGHSKIN
        | Abilities::IRONBARBS
        | Abilities::MUMMY
        | Abilities::WANDERINGSPIRIT
        | Abilities::GOOEY
        | Abilities::TANGLINGHAIR
        | Abilities::COTTONDOWN
        | Abilities::STATIC
        | Abilities::FLAMEBODY
        | Abilities::POISONPOINT
        | Abilities::CURSEDBODY
        | Abilities::AFTERMATH
        | Abilities::INNARDSOUT
        | Abilities::PERISHBODY
        | Abilities::LINGERINGAROMA
        | Abilities::GULPMISSILE => embedding[idx + 1] = 1.0,

        // On KO
        Abilities::MOXIE
        | Abilities::BEASTBOOST
        | Abilities::CHILLINGNEIGH
        | Abilities::GRIMNEIGH
        | Abilities::ASONEGLASTRIER
        | Abilities::ASONESPECTRIER => embedding[idx + 2] = 1.0,

        // End of turn
        Abilities::SPEEDBOOST
        | Abilities::MOODY
        | Abilities::BADDREAMS
        | Abilities::HARVEST
        | Abilities::HEALER
        | Abilities::RAINDISH
        | Abilities::ICEBODY
        | Abilities::SOLARPOWER
        | Abilities::HYDRATION
        | Abilities::SHEDSKIN
        | Abilities::REGENERATOR
        | Abilities::POISONHEAL => embedding[idx + 3] = 1.0,

        // When below certain HP
        Abilities::BERSERK
        | Abilities::EMERGENCYEXIT
        | Abilities::WIMPOUT
        | Abilities::TORRENT
        | Abilities::BLAZE
        | Abilities::OVERGROW
        | Abilities::SWARM => embedding[idx + 4] = 1.0,

        // Movement based
        Abilities::DANCER => embedding[idx + 5] = 1.0,

        // Type based
        Abilities::LEVITATE
        | Abilities::LIGHTNINGROD
        | Abilities::STORMDRAIN
        | Abilities::WATERABSORB
        | Abilities::VOLTABSORB
        | Abilities::MOTORDRIVE
        | Abilities::SAPSIPPER
        | Abilities::EARTHEATER
        | Abilities::WELLBAKEDBODY => embedding[idx + 6] = 1.0,

        // Form changing
        Abilities::SHIELDSDOWN
        | Abilities::SCHOOLING
        | Abilities::DISGUISE
        | Abilities::GULPMISSILE
        | Abilities::ZENMODE
        | Abilities::FORECAST
        | Abilities::POWERCONSTRUCT
        | Abilities::ICEFACE
        | Abilities::HUNGERSWITCH
        | Abilities::PROTEAN
        | Abilities::LIBERO => embedding[idx + 7] = 1.0,

        _ => {}
    }
    idx += TRIGGER_CONDITIONS_DIM;

    // Encode move effects
    match ability {
        // Priority modification
        Abilities::PRANKSTER => embedding[idx] = 1.0, // Priority for status moves
        Abilities::GALEWINGS => embedding[idx] = 0.5, // Priority for Flying moves when at full HP
        Abilities::TRIAGE => embedding[idx] = 0.5,    // Priority for healing moves

        // Move blocking
        Abilities::DAZZLING | Abilities::QUEENLYMAJESTY | Abilities::ARMORTAIL => {
            embedding[idx + 1] = 1.0
        } // Blocks priority moves
        Abilities::SOUNDPROOF => embedding[idx + 2] = 1.0, // Blocks sound moves
        Abilities::BULLETPROOF => embedding[idx + 3] = 1.0, // Blocks bullet moves
        Abilities::OVERCOAT => embedding[idx + 4] = 1.0,   // Blocks powder moves
        Abilities::GOODASGOLD => embedding[idx + 1] = 1.0, // Blocks status moves

        // Move modification
        Abilities::PROTEAN | Abilities::LIBERO => embedding[idx + 5] = 1.0, // Changes user's type
        Abilities::NORMALIZE => embedding[idx + 5] = 1.0, // Changes move type to Normal
        Abilities::AERILATE
        | Abilities::GALVANIZE
        | Abilities::PIXILATE
        | Abilities::REFRIGERATE => embedding[idx + 5] = 1.0, // Changes Normal-type moves

        // Critical hit modification
        Abilities::SUPERLUCK | Abilities::MERCILESS => embedding[idx + 6] = 1.0, // Increases critical hit ratio
        Abilities::BATTLEARMOR | Abilities::SHELLARMOR => embedding[idx + 6] = -1.0, // Prevents critical hits

        // Additional effects
        Abilities::SERENEGRACE => embedding[idx + 7] = 1.0, // Doubles secondary effect chance
        Abilities::SHEERFORCE => embedding[idx + 7] = -1.0, // Removes secondary effects but boosts power
        Abilities::MAGICBOUNCE => embedding[idx + 8] = 1.0, // Reflects status moves
        Abilities::SKILLLINK => embedding[idx + 9] = 1.0,   // Maximizes multi-hit moves

        // Move restrictions
        Abilities::GORILLATACTICS => embedding[idx + 8] = -1.0, // Locks into one move
        Abilities::TRUANT => embedding[idx + 8] = -1.0,         // Can only move every other turn

        _ => {}
    }
    idx += MOVE_EFFECTS_DIM;

    // Encode miscellaneous effects
    match ability {
        // Item interaction
        Abilities::MAGICIAN | Abilities::PICKPOCKET => embedding[idx] = 1.0, // Steals items
        Abilities::KLUTZ => embedding[idx] = -1.0,                           // Cannot use items
        Abilities::UNBURDEN => embedding[idx] = 0.5, // Speed boost when item is consumed
        Abilities::HARVEST => embedding[idx] = 0.5,  // Can recover consumed berries
        Abilities::RIPEN => embedding[idx] = 0.5,    // Doubles berry effects

        // Immunities to trapping
        Abilities::SHADOWTAG | Abilities::ARENATRAP => embedding[idx + 1] = -1.0, // Prevents escape
        Abilities::RUNAWAY => embedding[idx + 1] = 1.0, // Can always escape

        // Entry hazard interaction
        Abilities::LEVITATE => embedding[idx + 2] = 1.0, // Immune to Ground, including Spikes

        // Stat modification
        Abilities::CONTRARY => embedding[idx + 3] = 1.0, // Reverses stat changes
        Abilities::SIMPLE => embedding[idx + 3] = 1.0,   // Doubles stat changes

        // Weather immunity
        Abilities::AIRLOCK | Abilities::CLOUDNINE => embedding[idx + 4] = 1.0,

        // Prevents statuses
        Abilities::PASTELVEIL => embedding[idx + 5] = 1.0, // Prevents Poison
        Abilities::SWEETVEIL => embedding[idx + 5] = 1.0,  // Prevents Sleep

        // Copying abilities
        Abilities::POWEROFALCHEMY | Abilities::RECEIVER => embedding[idx + 6] = 1.0,
        Abilities::TRACE => embedding[idx + 6] = 1.0,

        // Utility effects
        Abilities::ILLUMINATE => embedding[idx + 7] = 1.0, // Increases wild encounter rate
        Abilities::HONEYGATHER => embedding[idx + 7] = 1.0, // Gathers honey
        Abilities::PICKUP => embedding[idx + 7] = 1.0,     // Picks up items

        // Immunity to specific move effects
        Abilities::CLEARBODY | Abilities::WHITESMOKE | Abilities::FULLMETALBODY => {
            embedding[idx + 8] = 1.0
        } // Prevents stat reduction

        // Special forms and transformations
        Abilities::DISGUISE | Abilities::ICEFACE => embedding[idx + 9] = 1.0, // Blocks one attack
        Abilities::IMPOSTER => embedding[idx + 9] = 1.0, // Transforms into opponent

        _ => {}
    }

    embedding
}

/// Create embeddings for all abilities
pub fn create_all_ability_embeddings() -> Vec<(Abilities, Array1<f32>)> {
    let mut embeddings = Vec::new();

    for ability in Abilities::iter() {
        let embedding = create_ability_embedding(&ability);
        embeddings.push((ability, embedding));
    }

    embeddings
}

pub fn create_all_item_embeddings() -> Vec<(Items, Array1<f32>)> {
    let mut embeddings = Vec::new();

    // Iterate through all item variants
    for item in Items::iter() {
        if item == Items::NONE {
            continue; // Skip NONE item
        }

        let embedding = create_item_embedding(item);
        embeddings.push((item, embedding));
    }

    embeddings
}
