use crate::data::conditions::PokemonSideCondition;
use crate::state::PokemonBoostableStat;
use crate::state::SideReference;
use crate::state::Terrain;
use crate::state::Weather;

use super::data::conditions::PokemonStatus;
use super::data::conditions::PokemonVolatileStatus;

// https://stackoverflow.com/questions/50686411/whats-the-usual-way-to-create-a-vector-of-different-structs
#[derive(Debug)]
pub enum Instruction {
    Switch(SwitchInstruction),
    VolatileStatus(VolatileStatusInstruction),
    ChangeStatus(ChangeStatusInstruction),
    Heal(HealInstruction),
    Damage(DamageInstruction),
    Boost(BoostInstruction),
    ChangeSideCondition(ChangeSideConditionInstruction),
    ChangeWeather(ChangeWeather),
    ChangeTerrain(ChangeTerrain),
}

#[derive(Debug)]
pub struct HealInstruction {
    pub side_ref: SideReference,
    pub heal_amount: i16,
}

#[derive(Debug)]
pub struct DamageInstruction {
    pub side_ref: SideReference,
    pub damage_amount: i16,
}

#[derive(Debug)]
pub struct SwitchInstruction {
    pub side_ref: SideReference,
    pub previous_index: usize,
    pub next_index: usize,
}

// pokemon_index is present because even reserve pokemon can have their status
// changed (i.e. healbell)
#[derive(Debug)]
pub struct ChangeStatusInstruction {
    pub side_ref: SideReference,
    pub pokemon_index: usize,
    pub old_status: PokemonStatus,
    pub new_status: PokemonStatus,
}

#[derive(Debug)]
pub struct VolatileStatusInstruction {
    pub side_ref: SideReference,
    pub volatile_status: PokemonVolatileStatus,
}

#[derive(Debug)]
pub struct BoostInstruction {
    pub side_ref: SideReference,
    pub stat: PokemonBoostableStat,
    pub amount: i8,
}

#[derive(Debug)]
pub struct ChangeSideConditionInstruction {
    pub side_ref: SideReference,
    pub side_condition: PokemonSideCondition,
    pub amount: i8,
}

#[derive(Debug)]
pub struct ChangeWeather {
    pub new_weather: Weather,
    pub new_weather_turns_remaining: i8,
    pub previous_weather: Weather,
    pub previous_weather_turns_remaining: i8,
}

#[derive(Debug)]
pub struct ChangeTerrain {
    pub new_terrain: Terrain,
    pub new_terrain_turns_remaining: i8,
    pub previous_terrain: Terrain,
    pub previous_terrain_turns_remaining: i8,
}