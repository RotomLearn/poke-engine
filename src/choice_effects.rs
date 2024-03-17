use crate::choices::{
    Boost, Choice, Choices, Heal, MoveCategory, MoveTarget, StatBoosts, VolatileStatus,
};
use crate::damage_calc::type_effectiveness_modifier;
use crate::instruction::{
    ApplyVolatileStatusInstruction, ChangeItemInstruction, ChangeSideConditionInstruction,
    ChangeTerrain, DamageInstruction, Instruction, SetSubstituteHealthInstruction,
    StateInstructions,
};
use crate::items::Items;
use crate::state::{
    PokemonBoostableStat, PokemonSideCondition, PokemonStatus, PokemonType, PokemonVolatileStatus,
    SideReference, State, Terrain, Weather,
};

pub fn modify_choice(
    state: &State,
    attacker_choice: &mut Choice,
    defender_choice: &Choice,
    attacking_side_ref: &SideReference,
) {
    let (attacking_side, defending_side) = state.get_both_sides_immutable(attacking_side_ref);
    match attacker_choice.move_id {
        Choices::BODYPRESS => {
            let attacker = attacking_side.get_active_immutable();
            attacker_choice.base_power *=
                attacker.calculate_boosted_stat(PokemonBoostableStat::Defense) as f32
                    / attacker.calculate_boosted_stat(PokemonBoostableStat::Attack) as f32;
        }
        Choices::BOLTBEAK | Choices::FISHIOUSREND => {
            if attacker_choice.first_move {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::CLANGOROUSSOUL => {
            let attacker = attacking_side.get_active_immutable();
            if attacker.hp > attacker.maxhp / 3 {
                attacker_choice.heal = Some(Heal {
                    target: MoveTarget::User,
                    amount: -0.33,
                });
                attacker_choice.boost = Some(Boost {
                    target: MoveTarget::User,
                    boosts: StatBoosts {
                        attack: 1,
                        defense: 1,
                        special_attack: 1,
                        special_defense: 1,
                        speed: 1,
                        accuracy: 0,
                    },
                });
            }
        }
        Choices::EXPANDINGFORCE => {
            if state.terrain.terrain_type == Terrain::PsychicTerrain {
                attacker_choice.base_power *= 1.5;
            }
        }
        Choices::FILLETAWAY => {
            let attacker = attacking_side.get_active_immutable();
            if attacker.hp > attacker.maxhp / 2 {
                attacker_choice.heal = Some(Heal {
                    target: MoveTarget::User,
                    amount: -0.5,
                });
                attacker_choice.boost = Some(Boost {
                    target: MoveTarget::User,
                    boosts: StatBoosts {
                        attack: 2,
                        defense: 0,
                        special_attack: 2,
                        special_defense: 0,
                        speed: 2,
                        accuracy: 0,
                    },
                });
            }
        }
        Choices::GROWTH => {
            if state.weather.weather_type == Weather::Sun {
                attacker_choice.boost = Some(Boost {
                    target: MoveTarget::User,
                    boosts: StatBoosts {
                        attack: 2,
                        defense: 0,
                        special_attack: 2,
                        special_defense: 0,
                        speed: 0,
                        accuracy: 0,
                    },
                });
            }
        }
        Choices::HEX => {
            if attacking_side.get_active_immutable().status != PokemonStatus::None {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::HYDROSTEAM => {
            if state.weather.weather_type == Weather::Sun {
                attacker_choice.base_power *= 3.0; // 1.5x for being in sun, 2x for cancelling out rain debuff
            }
        }
        Choices::MISTYEXPLOSION => {
            if state.terrain.terrain_type == Terrain::MistyTerrain {
                attacker_choice.base_power *= 1.5;
            }
        }
        Choices::MORNINGSUN | Choices::MOONLIGHT | Choices::SYNTHESIS => match state.weather.weather_type {
            Weather::Sun => {
                attacker_choice.heal = Some(Heal {
                    target: MoveTarget::User,
                    amount: 0.667,
                })
            }
            Weather::None => {}
            _ => {
                attacker_choice.heal = Some(Heal {
                    target: MoveTarget::User,
                    amount: 0.25,
                })
            }
        },
        Choices::NORETREAT => {
            if attacking_side
                .get_active_immutable()
                .volatile_statuses
                .contains(&PokemonVolatileStatus::NoRetreat)
            {
                attacker_choice.boost = None;
            }
        }
        Choices::POLTERGEIST => {
            if defending_side.get_active_immutable().item == Items::NONE {
                attacker_choice.base_power = 0.0;
            }
        }
        Choices::PROTECT => {
            if attacking_side.side_conditions.protect > 0 {
                // for now, the engine doesn't support consecutive protects
                // 2nd protect will always fail
                attacker_choice.volatile_status = None;
            }
        }
        Choices::PSYBLADE => {
            if state.terrain.terrain_type == Terrain::ElectricTerrain {
                attacker_choice.base_power *= 1.5;
            }
        }
        Choices::PURSUIT => {
            if defender_choice.category == MoveCategory::Switch {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::REVELATIONDANCE => {
            attacker_choice.move_type = attacking_side.get_active_immutable().types.0;
        }
        Choices::RISINGVOLTAGE => {
            if state.terrain.terrain_type == Terrain::ElectricTerrain {
                attacker_choice.base_power *= 1.5;
            }
        }
        Choices::SHOREUP => {
            if state.weather.weather_type == Weather::Sand {
                attacker_choice.heal = Some(Heal {
                    target: MoveTarget::User,
                    amount: 0.667,
                });
            }
        }
        Choices::STEELROLLER => {
            if state.terrain.terrain_type == Terrain::None {
                attacker_choice.base_power = 0.0;
            }
        }
        Choices::STRENGTHSAP => {
            attacker_choice.boost = Some(Boost {
                target: MoveTarget::Opponent,
                boosts: StatBoosts {
                    attack: -1,
                    defense: 0,
                    special_attack: 0,
                    special_defense: 0,
                    speed: 0,
                    accuracy: 0,
                },
            });
            let defender_attack = defending_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Attack);
            let attacker_maxhp = attacking_side.get_active_immutable().maxhp;
            attacker_choice.heal = Some(Heal {
                target: MoveTarget::User,
                amount: defender_attack as f32 / attacker_maxhp as f32,
            });
        }
        Choices::TERRAINPULSE => match state.terrain.terrain_type {
            Terrain::ElectricTerrain => {
                attacker_choice.move_type = PokemonType::Electric;
                attacker_choice.base_power *= 2.0;
            }
            Terrain::GrassyTerrain => {
                attacker_choice.move_type = PokemonType::Grass;
                attacker_choice.base_power *= 2.0;
            }
            Terrain::MistyTerrain => {
                attacker_choice.move_type = PokemonType::Fairy;
                attacker_choice.base_power *= 2.0;
            }
            Terrain::PsychicTerrain => {
                attacker_choice.move_type = PokemonType::Psychic;
                attacker_choice.base_power *= 2.0;
            }
            Terrain::None => {}
        },
        Choices::TOXIC => {
            if attacking_side
                .get_active_immutable()
                .has_type(&PokemonType::Poison)
            {
                attacker_choice.accuracy = 100.0;
            }
        }
        Choices::WEATHERBALL => match state.weather.weather_type {
            Weather::Sun | Weather::HarshSun => {
                attacker_choice.base_power = 100.0;
                attacker_choice.move_type = PokemonType::Fire;
            }
            Weather::Rain | Weather::HeavyRain => {
                attacker_choice.base_power = 100.0;
                attacker_choice.move_type = PokemonType::Water;
            }
            Weather::Sand => {
                attacker_choice.base_power = 100.0;
                attacker_choice.move_type = PokemonType::Rock;
            }
            Weather::Hail => {
                attacker_choice.base_power = 100.0;
                attacker_choice.move_type = PokemonType::Ice;
            }
            Weather::None => {}
        },
        Choices::SOLARBEAM => {
            if state.weather.weather_type == Weather::Sun
                || state.weather.weather_type == Weather::HarshSun
            {
                attacker_choice.flags.charge = false;
            }
        }
        Choices::BLIZZARD => {
            if state.weather.weather_type == Weather::Hail {
                attacker_choice.accuracy = 100.0;
            }
        }
        Choices::HURRICANE | Choices::THUNDER => {
            if state.weather.weather_type == Weather::Rain
                || state.weather.weather_type == Weather::HeavyRain
            {
                attacker_choice.accuracy = 100.0;
            } else if state.weather.weather_type == Weather::Sun
                || state.weather.weather_type == Weather::HarshSun
            {
                attacker_choice.accuracy = 50.0;
            }
        }
        Choices::KNOCKOFF => {
            if defending_side.get_active_immutable().item_can_be_removed() {
                attacker_choice.base_power *= 1.5;
            }
        }
        Choices::ACROBATICS => {
            if attacking_side.get_active_immutable().item == Items::NONE {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::FOCUSPUNCH => {
            if attacker_choice.first_move || defender_choice.category != MoveCategory::Status {
                attacker_choice.remove_all_effects();
            }
        }
        Choices::ELECTROBALL => {
            let attacker_speed = attacking_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Speed);
            let defender_speed = defending_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Speed);
            let speed_ratio = attacker_speed as f32 / defender_speed as f32;
            if speed_ratio >= 4.0 {
                attacker_choice.base_power = 150.0;
            } else if speed_ratio >= 3.0 {
                attacker_choice.base_power = 120.0;
            } else if speed_ratio >= 2.0 {
                attacker_choice.base_power = 80.0;
            } else if speed_ratio >= 1.0 {
                attacker_choice.base_power = 60.0;
            } else {
                attacker_choice.base_power = 40.0;
            }
        }
        Choices::GYROBALL => {
            let attacker_speed = attacking_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Speed);
            let defender_speed = defending_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Speed);

            attacker_choice.base_power =
                ((25.0 * defender_speed as f32 / attacker_speed as f32) + 1.0).min(150.0);
        }
        Choices::AVALANCHE => {
            if !attacker_choice.first_move
                && (defender_choice.category == MoveCategory::Physical
                    || defender_choice.category == MoveCategory::Special)
            {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::FACADE => {
            if attacking_side.get_active_immutable().status != PokemonStatus::None {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::PSYSHOCK | Choices::SECRETSWORD | Choices::PSYSTRIKE => {
            let defender = defending_side.get_active_immutable();
            let defender_defense = defender.calculate_boosted_stat(PokemonBoostableStat::Defense);
            let defender_special_defense =
                defender.calculate_boosted_stat(PokemonBoostableStat::SpecialDefense);
            attacker_choice.base_power *= defender_special_defense as f32 / defender_defense as f32
        }
        Choices::STOREDPOWER => {
            let attacking_pokemon = attacking_side.get_active_immutable();
            let total_boosts = attacking_pokemon.attack_boost
                + attacking_pokemon.defense_boost
                + attacking_pokemon.special_attack_boost
                + attacking_pokemon.special_defense_boost
                + attacking_pokemon.speed_boost
                + attacking_pokemon.accuracy_boost
                + attacking_pokemon.evasion_boost;
            if total_boosts > 0 {
                attacker_choice.base_power *= total_boosts as f32;
            }
        }
        Choices::FOULPLAY => {
            let defender = defending_side.get_active_immutable();
            let defender_attack = defender.calculate_boosted_stat(PokemonBoostableStat::Attack);
            let attacker_attack = attacking_side
                .get_active_immutable()
                .calculate_boosted_stat(PokemonBoostableStat::Attack);
            attacker_choice.base_power = defender_attack as f32 / attacker_attack as f32;
        }
        Choices::BARBBARRAGE => {
            let defending_pkmn_status = defending_side.get_active_immutable().status;
            if defending_pkmn_status == PokemonStatus::Poison
                || defending_pkmn_status == PokemonStatus::Toxic
            {
                attacker_choice.base_power *= 2.0;
            }
        }
        Choices::FREEZEDRY => {
            if defending_side
                .get_active_immutable()
                .has_type(&PokemonType::Water)
            {
                attacker_choice.base_power *= 4.0; // 2x for being super effective, 2x for nullifying water resistance
            }
        }
        Choices::ERUPTION | Choices::WATERSPOUT => {
            let attacker = attacking_side.get_active_immutable();
            let hp_ratio = attacker.hp as f32 / attacker.maxhp as f32;
            attacker_choice.base_power *= hp_ratio;
        }
        Choices::SUCKERPUNCH => {
            if !attacker_choice.first_move || defender_choice.category == MoveCategory::Status {
                attacker_choice.base_power = 0.0;
            }
        }
        Choices::COLLISIONCOURSE => {
            let defender_active = defending_side.get_active_immutable();
            if type_effectiveness_modifier(&attacker_choice.move_type, &defender_active.types) > 1.0
            {
                attacker_choice.base_power *= 1.3;
            }
        }
        _ => {}
    }
}

pub fn choice_after_damage_hit(
    state: &mut State,
    choice: &Choice,
    attacking_side_ref: &SideReference,
    instructions: &mut StateInstructions,
) {
    let (_attacking_side, defending_side) = state.get_both_sides(attacking_side_ref);
    match choice.move_id {
        Choices::KNOCKOFF => {
            let defender_active = defending_side.get_active();
            if defender_active.item_can_be_removed() {
                let instruction = Instruction::ChangeItem(ChangeItemInstruction {
                    side_ref: attacking_side_ref.get_other_side(),
                    current_item: defender_active.item,
                    new_item: Items::NONE,
                });
                instructions.instruction_list.push(instruction);
                defender_active.item = Items::NONE;
            }
        }
        _ => {}
    }
}

pub fn choice_hazard_clear(
    state: &mut State,
    choice: &Choice,
    attacking_side_ref: &SideReference,
    instructions: &mut StateInstructions,
) {
    let (attacking_side, _defending_side) = state.get_both_sides(attacking_side_ref);
    match choice.move_id {
        Choices::COURTCHANGE => {
            let mut instruction_list = vec![];
            let courtchange_swaps = [
                PokemonSideCondition::Stealthrock,
                PokemonSideCondition::Spikes,
                PokemonSideCondition::ToxicSpikes,
                PokemonSideCondition::StickyWeb,
                PokemonSideCondition::Reflect,
                PokemonSideCondition::LightScreen,
                PokemonSideCondition::AuroraVeil,
                PokemonSideCondition::Tailwind,
            ];

            for side in [SideReference::SideOne, SideReference::SideTwo] {
                for side_condition in courtchange_swaps {
                    let side_condition_num = state
                        .get_side_immutable(&side)
                        .get_side_condition(side_condition);
                    if side_condition_num > 0 {
                        instruction_list.push(Instruction::ChangeSideCondition(
                            ChangeSideConditionInstruction {
                                side_ref: side,
                                side_condition: side_condition,
                                amount: -1 * side_condition_num,
                            },
                        ));
                        instruction_list.push(Instruction::ChangeSideCondition(
                            ChangeSideConditionInstruction {
                                side_ref: side.get_other_side(),
                                side_condition: side_condition,
                                amount: side_condition_num,
                            },
                        ));
                    }
                }
            }
            state.apply_instructions(&instruction_list);
            for i in instruction_list {
                instructions.instruction_list.push(i)
            }
        }
        Choices::DEFOG => {
            if state.terrain.terrain_type != Terrain::None {
                instructions
                    .instruction_list
                    .push(Instruction::ChangeTerrain(ChangeTerrain {
                        new_terrain: Terrain::None,
                        new_terrain_turns_remaining: 0,
                        previous_terrain: state.terrain.terrain_type,
                        previous_terrain_turns_remaining: state.terrain.turns_remaining,
                    }));
                state.terrain.terrain_type = Terrain::None;
                state.terrain.turns_remaining = 0;
            }
            let side_condition_clears = [
                PokemonSideCondition::Stealthrock,
                PokemonSideCondition::Spikes,
                PokemonSideCondition::ToxicSpikes,
                PokemonSideCondition::StickyWeb,
                PokemonSideCondition::Reflect,
                PokemonSideCondition::LightScreen,
                PokemonSideCondition::AuroraVeil,
            ];

            for side in [SideReference::SideOne, SideReference::SideTwo] {
                for side_condition in side_condition_clears {
                    let side_condition_num = state
                        .get_side_immutable(&side)
                        .get_side_condition(side_condition);
                    if side_condition_num > 0 {
                        let i = Instruction::ChangeSideCondition(ChangeSideConditionInstruction {
                            side_ref: side,
                            side_condition: side_condition,
                            amount: -1 * side_condition_num,
                        });
                        state.apply_one_instruction(&i);
                        instructions.instruction_list.push(i)
                    }
                }
            }
        }
        Choices::RAPIDSPIN => {
            if attacking_side.side_conditions.stealth_rock > 0 {
                instructions
                    .instruction_list
                    .push(Instruction::ChangeSideCondition(
                        ChangeSideConditionInstruction {
                            side_ref: *attacking_side_ref,
                            side_condition: PokemonSideCondition::Stealthrock,
                            amount: -1 * attacking_side.side_conditions.stealth_rock,
                        },
                    ));
                attacking_side.side_conditions.stealth_rock = 0;
            }
            if attacking_side.side_conditions.spikes > 0 {
                instructions
                    .instruction_list
                    .push(Instruction::ChangeSideCondition(
                        ChangeSideConditionInstruction {
                            side_ref: *attacking_side_ref,
                            side_condition: PokemonSideCondition::Spikes,
                            amount: -1 * attacking_side.side_conditions.spikes,
                        },
                    ));
                attacking_side.side_conditions.spikes = 0;
            }
            if attacking_side.side_conditions.toxic_spikes > 0 {
                instructions
                    .instruction_list
                    .push(Instruction::ChangeSideCondition(
                        ChangeSideConditionInstruction {
                            side_ref: *attacking_side_ref,
                            side_condition: PokemonSideCondition::ToxicSpikes,
                            amount: -1 * attacking_side.side_conditions.toxic_spikes,
                        },
                    ));
                attacking_side.side_conditions.toxic_spikes = 0;
            }
            if attacking_side.side_conditions.sticky_web > 0 {
                instructions
                    .instruction_list
                    .push(Instruction::ChangeSideCondition(
                        ChangeSideConditionInstruction {
                            side_ref: *attacking_side_ref,
                            side_condition: PokemonSideCondition::StickyWeb,
                            amount: -1 * attacking_side.side_conditions.sticky_web,
                        },
                    ));
                attacking_side.side_conditions.sticky_web = 0;
            }
        }
        _ => {}
    }
}

pub fn choice_special_effect(
    state: &mut State,
    choice: &Choice,
    attacking_side_ref: &SideReference,
    instructions: &mut StateInstructions,
) {
    let (attacking_side, defending_side) = state.get_both_sides(attacking_side_ref);
    match choice.move_id {
        Choices::PAINSPLIT => {
            let target_hp = (attacking_side.get_active_immutable().hp
                + defending_side.get_active_immutable().hp)
                / 2;
            instructions
                .instruction_list
                .push(Instruction::Damage(DamageInstruction {
                    side_ref: *attacking_side_ref,
                    damage_amount: attacking_side.get_active_immutable().hp - target_hp,
                }));
            instructions
                .instruction_list
                .push(Instruction::Damage(DamageInstruction {
                    side_ref: attacking_side_ref.get_other_side(),
                    damage_amount: defending_side.get_active_immutable().hp - target_hp,
                }));

            attacking_side.get_active().hp = target_hp;
            defending_side.get_active().hp = target_hp;
        }
        Choices::SUBSTITUTE => {
            let active_pkmn = attacking_side.get_active();
            let sub_target_health = active_pkmn.maxhp / 4;
            if active_pkmn.hp > sub_target_health {
                let damage_instruction = Instruction::Damage(DamageInstruction {
                    side_ref: attacking_side_ref.clone(),
                    damage_amount: sub_target_health,
                });
                let set_sub_health_instruction =
                    Instruction::SetSubstituteHealth(SetSubstituteHealthInstruction {
                        side_ref: attacking_side_ref.clone(),
                        new_health: sub_target_health,
                        old_health: active_pkmn.substitute_health,
                    });
                let apply_vs_instruction =
                    Instruction::ApplyVolatileStatus(ApplyVolatileStatusInstruction {
                        side_ref: attacking_side_ref.clone(),
                        volatile_status: PokemonVolatileStatus::Substitute,
                    });
                active_pkmn.hp -= sub_target_health;
                active_pkmn.substitute_health = sub_target_health;
                active_pkmn
                    .volatile_statuses
                    .insert(PokemonVolatileStatus::Substitute);
                instructions.instruction_list.push(damage_instruction);
                instructions
                    .instruction_list
                    .push(set_sub_health_instruction);
                instructions.instruction_list.push(apply_vs_instruction);
            }
        }
        Choices::TRICK => {
            let attacker = attacking_side.get_active();
            let defender = defending_side.get_active();
            let attacker_item = attacker.item;
            let defender_item = defender.item;
            if attacker_item == defender_item {
                return;
            }
            let change_attacker_item_instruction = Instruction::ChangeItem(ChangeItemInstruction {
                side_ref: *attacking_side_ref,
                current_item: attacker_item,
                new_item: defender_item,
            });
            let change_defender_item_instruction = Instruction::ChangeItem(ChangeItemInstruction {
                side_ref: attacking_side_ref.get_other_side(),
                current_item: defender_item,
                new_item: attacker_item,
            });
            attacker.item = defender_item;
            defender.item = attacker_item;
            instructions
                .instruction_list
                .push(change_attacker_item_instruction);
            instructions
                .instruction_list
                .push(change_defender_item_instruction);
        }
        _ => {}
    }
}

pub fn charge_choice_to_volatile(choice: &Choices) -> PokemonVolatileStatus {
    return match choice {
        Choices::BOUNCE => PokemonVolatileStatus::Bounce,
        Choices::DIG => PokemonVolatileStatus::Dig,
        Choices::DIVE => PokemonVolatileStatus::Dive,
        Choices::FLY => PokemonVolatileStatus::Fly,
        Choices::FREEZESHOCK => PokemonVolatileStatus::Freezeshock,
        Choices::GEOMANCY => PokemonVolatileStatus::Geomancy,
        Choices::ICEBURN => PokemonVolatileStatus::IceBurn,
        Choices::METEORBEAM => PokemonVolatileStatus::MeteorBeam,
        Choices::PHANTOMFORCE => PokemonVolatileStatus::PhantomForce,
        Choices::RAZORWIND => PokemonVolatileStatus::RazorWind,
        Choices::SHADOWFORCE => PokemonVolatileStatus::ShadowForce,
        Choices::SKULLBASH => PokemonVolatileStatus::SkullBash,
        Choices::SKYATTACK => PokemonVolatileStatus::SkyAttack,
        Choices::SKYDROP => PokemonVolatileStatus::SkyDrop,
        Choices::SOLARBEAM => PokemonVolatileStatus::SolarBeam,
        Choices::SOLARBLADE => PokemonVolatileStatus::SolarBlade,
        _ => {
            panic!("Invalid choice for charge: {:?}", choice)
        }
    };
}