#![allow(unused_variables)]
use std::cmp;
use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::choices::{
    Choice, Effect, Heal, MoveCategory, MoveTarget, Secondary, StatBoosts, VolatileStatus,
};
use crate::damage_calc::type_effectiveness_modifier;
use crate::generate_instructions::get_boost_instruction;
use crate::instruction::{
    BoostInstruction, ChangeStatusInstruction, ChangeType, HealInstruction, Instruction,
};
use crate::state::{PokemonBoostableStat, PokemonType, Terrain};
use crate::state::{PokemonStatus, State};
use crate::state::{PokemonVolatileStatus, SideReference, Weather};

type ModifyAttackBeingUsed = fn(&State, &mut Choice, &Choice, &SideReference);
type ModifyAttackAgainst = fn(&State, &mut Choice, &Choice, &SideReference);
type AbilityBeforeMove = fn(&State, &Choice, &SideReference) -> Vec<Instruction>;
type AbilityAfterDamageHit = fn(&State, &Choice, &SideReference, i16) -> Vec<Instruction>;
type AbilityOnSwitchOut = fn(&State, &SideReference) -> Vec<Instruction>;
type AbilityOnSwitchIn = fn(&State, &SideReference) -> Vec<Instruction>;
type AbilityEndOfTurn = fn(&State, &SideReference) -> Vec<Instruction>;

pub struct Ability {
    pub modify_attack_being_used: Option<ModifyAttackBeingUsed>,
    pub modify_attack_against: Option<ModifyAttackAgainst>,
    pub before_move: Option<AbilityBeforeMove>,
    pub after_damage_hit: Option<AbilityAfterDamageHit>,
    pub on_switch_out: Option<AbilityOnSwitchOut>,
    pub on_switch_in: Option<AbilityOnSwitchIn>,
    pub end_of_turn: Option<AbilityEndOfTurn>,
}

lazy_static! {
    pub static ref ABILITIES: HashMap<String, Ability> = {
        let mut abilities: HashMap<String, Ability> = HashMap::new();
        abilities.insert(
            "ripen".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "tangledfeet".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "dragonsmaw".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Dragon {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "clearbody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "galvanize".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Normal {
                            attacking_choice.move_type = PokemonType::Electric;
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "vitalspirit".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "aerilate".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Normal {
                            attacking_choice.move_type = PokemonType::Flying;
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "defiant".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "cutecharm".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "neuroforce".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if type_effectiveness_modifier(
                            &attacking_choice.move_type,
                            &state
                                .get_side_immutable(&attacking_side.get_other_side())
                                .get_active_immutable()
                                .types,
                        ) > 1.0
                        {
                            attacking_choice.base_power *= 1.25;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "soundproof".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "rkssystem".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "poisonpoint".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.contact {
                            attacking_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 30.0,
                                    target: MoveTarget::Opponent,
                                    effect: Effect::Status(PokemonStatus::Poison),
                                }
                            )
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "stakeout".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if defender_choice.category == MoveCategory::Switch {
                            attacking_choice.base_power *= 2.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "unnerve".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "rockhead".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "aurabreak".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "mimicry".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "bulletproof".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.flags.bullet {
                            attacker_choice.accuracy = 0.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "powerofalchemy".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "technician".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.base_power <= 60.0 {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "multiscale".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "arenatrap".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "battlebond".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "disguise".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "earlybird".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "lightningrod".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "magician".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "refrigerate".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Normal {
                            attacking_choice.move_type = PokemonType::Ice;
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "friendguard".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "noability".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "gulpmissile".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "powerconstruct".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "forecast".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "prankster".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "protean".to_string(),
            Ability {
                before_move: Some(|state: &State, choice: &Choice, side_ref: &SideReference| {
                    let active_pkmn = state.get_side_immutable(side_ref).get_active_immutable();
                    if !active_pkmn.has_type(&choice.move_type) {
                        return vec![Instruction::ChangeType(ChangeType {
                            side_ref: *side_ref,
                            new_types: (choice.move_type, PokemonType::Typeless),
                            old_types: active_pkmn.types,
                        })];
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "asoneglastrier".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "shadowtag".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "skilllink".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "intrepidsword".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "soulheart".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "swiftswim".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "eartheater".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Ground {
                            attacker_choice.base_power = 0.0;
                            attacker_choice.heal = Some(Heal {
                                target: MoveTarget::Opponent,
                                amount: 0.25
                            });
                            attacker_choice.category = MoveCategory::Status;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "superluck".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "supremeoverlord".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let mut boost_amount = 1.0;
                        let side = state.get_side_immutable(attacking_side);
                        for (index, pkmn) in side.pokemon.iter().enumerate() {
                            if pkmn.hp <= 0 && index != side.active_index {
                                boost_amount += 0.1;
                            }
                        }
                        attacking_choice.base_power *= boost_amount;
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "insomnia".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "dancer".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "steamengine".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "angerpoint".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "contrary".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "magmaarmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "hungerswitch".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "receiver".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "zenmode".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "emergencyexit".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "illusion".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "weakarmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "drought".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "innardsout".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "shieldsdown".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "adaptability".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if state
                            .get_side_immutable(attacking_side)
                            .get_active_immutable()
                            .has_type(&attacking_choice.move_type)
                        {
                            attacking_choice.base_power *= 4.0 / 3.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "corrosion".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "longreach".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        attacking_choice.flags.contact = false;
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "purepower".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.category == MoveCategory::Physical {
                            attacking_choice.base_power *= 2.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "tintedlens".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if type_effectiveness_modifier(
                            &attacking_choice.move_type,
                            &state
                                .get_side_immutable(&attacking_side.get_other_side())
                                .get_active_immutable()
                                .types,
                        ) < 1.0
                        {
                            attacking_choice.base_power *= 2.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "queenlymajesty".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "desolateland".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "moxie".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sapsipper".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "slushrush".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "bigpecks".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "stall".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "whitesmoke".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "flareboost".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if state.get_side_immutable(attacking_side).get_active_immutable().status == PokemonStatus::Burn {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "shadowshield".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "liquidvoice".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.sound {
                            attacking_choice.move_type = PokemonType::Water;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "mistysurge".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "multitype".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "noguard".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        attacking_choice.accuracy = 100.0
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "torrent".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pokemon = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_choice.move_type == PokemonType::Water && attacking_pokemon.hp < attacking_pokemon.maxhp / 3 {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "deltastream".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "klutz".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "libero".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "serenegrace".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if let Some(secondaries) = &mut attacking_choice.secondaries {
                            for secondary in secondaries.iter_mut() {
                                secondary.chance *= 2.0;
                            }
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "cursedbody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "unaware".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "lightmetal".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "marvelscale".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "telepathy".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "quickdraw".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "hypercutter".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "symbiosis".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "plus".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "mirrorarmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "pastelveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "toughclaws".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.contact {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "effectspore".to_string(),
            Ability {
                modify_attack_against: Some(
                    |_state, attacker_choice: &mut Choice, _defender_choice, _attacking_side| {
                        if attacker_choice.flags.contact {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 9.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Status(PokemonStatus::Poison),
                                }
                            );
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 10.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Status(PokemonStatus::Paralyze),
                                }
                            );
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 11.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Status(PokemonStatus::Sleep),
                                }
                            );
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "mummy".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "baddreams".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "magicguard".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sandstream".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "powerspot".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "flamebody".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if state.move_makes_contact(&attacker_choice, attacking_side) {
                            let burn_secondary = Secondary {
                                chance: 30.0,
                                target: MoveTarget::User,
                                effect: Effect::Status(PokemonStatus::Burn),
                            };

                            if attacker_choice.secondaries.is_none() {
                                attacker_choice.secondaries = Some(vec![burn_secondary]);
                            } else {
                                attacker_choice
                                    .secondaries
                                    .as_mut()
                                    .unwrap()
                                    .push(burn_secondary);
                            }
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "reckless".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.crash.is_some() {
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "pressure".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "gooey".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.flags.contact {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 100.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Boost(
                                        StatBoosts {
                                            attack: 0,
                                            defense: 0,
                                            special_attack: 0,
                                            special_defense: 0,
                                            speed: -1,
                                            accuracy: 0,
                                        }
                                    ),
                                }
                            )
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "immunity".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "leafguard".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "hugepower".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.category == MoveCategory::Physical {
                            attacking_choice.base_power *= 2.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "solarpower".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if state.weather_is_active(&Weather::Sun) {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "schooling".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "motordrive".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "anticipation".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "merciless".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "trace".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "naturalcure".to_string(),
            Ability {
                on_switch_out: Some(|state: &State, side_reference: &SideReference| {
                    let side = state.get_side_immutable(side_reference);
                    if side.get_active_immutable().status != PokemonStatus::None {
                        return vec![Instruction::ChangeStatus(ChangeStatusInstruction {
                            side_ref: *side_reference,
                            pokemon_index: side.active_index,
                            old_status: side.get_active_immutable().status,
                            new_status: PokemonStatus::None,
                        })];
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "harvest".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "suctioncups".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "iceface".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "roughskin".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "wonderguard".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "waterveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "fairyaura".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Fairy {
                            attacking_choice.base_power *= 1.33;
                        }
                    },
                ),
                modify_attack_against: Some(
                    |_state, attacker_choice: &mut Choice, _defender_choice, _attacking_side| {
                        if attacker_choice.move_type == PokemonType::Fairy {
                            attacker_choice.base_power *= 1.33;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "sandspit".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "intimidate".to_string(),
            Ability {
                on_switch_in: Some(|state: &State, side_ref: &SideReference| {
                    if let Some(boost_instruction) = get_boost_instruction(
                        state,
                        &PokemonBoostableStat::Attack,
                        &-1,
                        side_ref,
                        &side_ref.get_other_side(),
                    ) {
                        return vec![boost_instruction];
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "dauntlessshield".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "aromaveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "airlock".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "normalize".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        attacking_choice.move_type = PokemonType::Normal;
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "darkaura".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Dark {
                            attacking_choice.base_power *= 1.33;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "victorystar".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        attacking_choice.accuracy *= 1.1;
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "grassysurge".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sturdy".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "pickpocket".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "electricsurge".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "runaway".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "oblivious".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "surgesurfer".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "levitate".to_string(),
            Ability {
                modify_attack_against: Some(
                    |_state, attacker_choice: &mut Choice, _defender_choice, _attacking_side| {
                        if attacker_choice.move_type == PokemonType::Ground
                            && attacker_choice.target == MoveTarget::Opponent
                            && attacker_choice.move_id != "thousandarrows"
                        {
                            attacker_choice.base_power = 0.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "asonespectrier".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "pickup".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "icebody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "curiousmedicine".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "flowerveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "static".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if state.move_makes_contact(&attacker_choice, attacking_side) {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 30.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Status(PokemonStatus::Paralyze),
                                }
                            )
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "wonderskin".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "overgrow".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pokemon = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_choice.move_type == PokemonType::Grass && attacking_pokemon.hp < attacking_pokemon.maxhp / 3 {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "propellertail".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "thickfat".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "gluttony".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "keeneye".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "mountaineer".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "flashfire".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Fire {
                            attacker_choice.base_power = 0.0;
                            attacker_choice.volatile_status = Some(VolatileStatus {
                                target: MoveTarget::Opponent,
                                volatile_status: PokemonVolatileStatus::FlashFire,
                            });
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "compoundeyes".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        attacking_choice.accuracy *= 1.3;
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "steelworker".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if defender_choice.move_type == PokemonType::Steel {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "comatose".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "ballfetch".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "dazzling".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.priority > 0 {
                            attacker_choice.accuracy = 0.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "download".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "transistor".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Electric {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "moldbreaker".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "liquidooze".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "poisonheal".to_string(),
            Ability {
                end_of_turn: Some(|state: &State, side_ref: &SideReference| {
                    let attacker = state.get_side_immutable(side_ref).get_active_immutable();
                    if attacker.hp < attacker.maxhp
                        && (attacker.status == PokemonStatus::Poison
                            || attacker.status == PokemonStatus::Toxic)
                    {
                        return vec![Instruction::Heal(HealInstruction {
                            side_ref: side_ref.clone(),
                            heal_amount: cmp::min(attacker.maxhp / 8, attacker.maxhp - attacker.hp),
                        })];
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "prismarmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sniper".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "stench".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let mut already_flinches = false;
                        if let Some(secondaries) = &mut attacking_choice.secondaries {
                            for secondary in secondaries.iter() {
                                if secondary.effect == Effect::VolatileStatus(PokemonVolatileStatus::Flinch) {
                                    already_flinches = true;
                                }
                            }
                        }
                        if !already_flinches {
                            attacking_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 10.0,
                                    target: MoveTarget::Opponent,
                                    effect: Effect::VolatileStatus(PokemonVolatileStatus::Flinch),
                                }
                            )
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "competitive".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "swarm".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pokemon = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_choice.move_type == PokemonType::Bug && attacking_pokemon.hp < attacking_pokemon.maxhp / 3 {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "stalwart".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "illuminate".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "turboblaze".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "gorillatactics".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.category == MoveCategory::Physical {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "speedboost".to_string(),
            Ability {
                end_of_turn: Some(|state: &State, side_ref: &SideReference| {
                    let attacker = state.get_side_immutable(side_ref).get_active_immutable();
                    if attacker.speed_boost < 6 {
                        return vec![Instruction::Boost(BoostInstruction {
                            side_ref: side_ref.clone(),
                            stat: PokemonBoostableStat::Speed,
                            amount: 1,
                        })];
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "heatproof".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Fire {
                            attacker_choice.base_power *= 0.5 ;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "snowcloak".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "teravolt".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "chillingneigh".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "shielddust".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "rivalry".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "primordialsea".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "screencleaner".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "magnetpull".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "honeygather".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "cottondown".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "grasspelt".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if state.terrain_is_active(&Terrain::GrassyTerrain) && attacker_choice.category == MoveCategory::Physical {
                            attacker_choice.base_power /= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "battlearmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "beastboost".to_string(),
            Ability {
                after_damage_hit: Some(|state, _, attacking_side, damage_dealt| {
                    let (attacker_side, defender_side) =
                        state.get_both_sides_immutable(attacking_side);
                    if defender_side.get_active_immutable().hp == damage_dealt {
                        if let Some(boost_instruction) = get_boost_instruction(
                            state,
                            &attacker_side
                                .get_active_immutable()
                                .calculate_highest_stat(),
                            &1,
                            attacking_side,
                            attacking_side,
                        ) {
                            return vec![boost_instruction];
                        }
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "berserk".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "minus".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "raindish".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "synchronize".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "filter".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if type_effectiveness_modifier(
                            &attacker_choice.move_type,
                            &state
                                .get_side_immutable(&attacking_side.get_other_side())
                                .get_active_immutable()
                                .types,
                        ) > 1.0
                        {
                            attacker_choice.base_power *= 0.75;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "truant".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "furcoat".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.category == MoveCategory::Physical {
                            attacker_choice.base_power *= 0.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "fullmetalbody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "regenerator".to_string(),
            Ability {
                on_switch_out: Some(|state: &State, side_ref: &SideReference| {
                    let switching_out_pkmn =
                        state.get_side_immutable(side_ref).get_active_immutable();
                    let hp_recovered = cmp::min(
                        switching_out_pkmn.maxhp / 3,
                        switching_out_pkmn.maxhp - switching_out_pkmn.hp,
                    );

                    if hp_recovered > 0 && switching_out_pkmn.hp > 0 {
                        return vec![Instruction::Heal(HealInstruction {
                            side_ref: *side_ref,
                            heal_amount: hp_recovered,
                        })];
                    }

                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "forewarn".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "ironbarbs".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.flags.contact {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 100.0,
                                    target: MoveTarget::User,
                                    effect: Effect::Heal(-0.125),
                                }
                            );
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "stamina".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sandrush".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "colorchange".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "blaze".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pokemon = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_choice.move_type == PokemonType::Fire && attacking_pokemon.hp < attacking_pokemon.maxhp / 3 {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "analytic".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if !attacking_choice.first_move {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "tanglinghair".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "cloudnine".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "steelyspirit".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "quickfeet".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "magicbounce".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "megalauncher".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.pulse {
                            attacking_choice.base_power *= 1.5;
                        };
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "heavymetal".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "stormdrain".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "pixilate".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Normal {
                            attacking_choice.move_type = PokemonType::Fairy;
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "watercompaction".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "justified".to_string(),
            Ability {
                modify_attack_against: Some(
                    |_state, attacker_choice: &mut Choice, _defender_choice, _attacking_side| {
                        if attacker_choice.move_type == PokemonType::Dark {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 100.0,
                                    target: MoveTarget::Opponent,
                                    effect: Effect::Boost(
                                        StatBoosts {
                                            attack: 1,
                                            defense: 0,
                                            special_attack: 0,
                                            special_defense: 0,
                                            speed: 0,
                                            accuracy: 0,
                                        }
                                    ),
                                }
                            )
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "slowstart".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "snowwarning".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "flowergift".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "shedskin".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "wimpout".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "icescales".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.category == MoveCategory::Special {
                            attacker_choice.base_power *= 0.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "infiltrator".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "limber".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "psychicsurge".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "defeatist".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pokemon = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_pokemon.hp < attacking_pokemon.maxhp / 2 {
                            attacking_choice.base_power *= 0.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "waterabsorb".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Water {
                            attacker_choice.base_power = 0.0;
                            attacker_choice.heal = Some(Heal {
                                target: MoveTarget::Opponent,
                                amount: 0.25
                            });
                            attacker_choice.category = MoveCategory::Status;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "imposter".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "dryskin".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Water {
                            attacker_choice.base_power = 0.0;
                            attacker_choice.heal = Some(Heal {
                                target: MoveTarget::Opponent,
                                amount: 0.25
                            });
                            attacker_choice.category = MoveCategory::Status;
                        } else if attacker_choice.move_type == PokemonType::Fire {
                            attacker_choice.base_power *= 1.25;
                        }
                    },
                ),
                end_of_turn: Some(|state: &State, side_ref: &SideReference| {
                    if state.weather_is_active(&Weather::Rain) {
                        let active_pkmn = state.get_side_immutable(side_ref).get_active_immutable();

                        if active_pkmn.hp < active_pkmn.maxhp {
                            return vec![Instruction::Heal(HealInstruction {
                                side_ref: side_ref.clone(),
                                heal_amount: cmp::min(active_pkmn.maxhp / 8, active_pkmn.maxhp - active_pkmn.hp),
                            })];
                        }
                    }
                    return vec![];
                }),
                ..Default::default()
            },
        );
        abilities.insert(
            "fluffy".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.flags.contact {
                            attacker_choice.base_power *= 0.5;
                        }
                        if attacker_choice.move_type == PokemonType::Fire {
                            attacker_choice.base_power *= 2.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "unburden".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "cheekpouch".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "stancechange".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "moody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "rockypayload".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.move_type == PokemonType::Rock {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "punkrock".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.sound {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "sandveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "parentalbond".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "strongjaw".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.bite {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "battery".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.category == MoveCategory::Special {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "healer".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "steadfast".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "damp".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if ["selfdestruct", "explosion", "mindblown", "mistyexplosion"].contains(&attacker_choice.move_id.as_str()) {
                            attacker_choice.accuracy = 0.0;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "perishbody".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "triage".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sheerforce".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.secondaries.is_some() {
                            attacking_choice.base_power *= 1.3;
                            attacking_choice.secondaries = None
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "owntempo".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "frisk".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "voltabsorb".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "galewings".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "aftermath".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "stickyhold".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "grimneigh".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "ironfist".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.punch {
                            attacking_choice.base_power *= 1.2;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "rebound".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "unseenfist".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.flags.contact {
                            attacking_choice.flags.protect = false
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "solidrock".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "hustle".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if attacking_choice.category == MoveCategory::Physical {
                            attacking_choice.base_power *= 1.5;
                            attacking_choice.accuracy *= 0.80
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "hydration".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "scrappy".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if state.get_side_immutable(&attacking_side.get_other_side()).get_active_immutable().has_type(&PokemonType::Ghost) {
                            // Technically wrong, come back to this later
                            attacking_choice.move_type = PokemonType::Typeless;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "overcoat".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "neutralizinggas".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sweetveil".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "drizzle".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "innerfocus".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "poisontouch".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "wanderingspirit".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "guts".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let attacking_pkmn = state.get_side_immutable(attacking_side).get_active_immutable();
                        if attacking_pkmn.status != PokemonStatus::None {
                            attacking_choice.base_power *= 1.5;

                            // not the right place to put this, but good enough
                            if attacking_pkmn.status == PokemonStatus::Burn {
                                attacking_choice.base_power *= 2.0;
                            }
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "shellarmor".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "rattled".to_string(),
            Ability {
                modify_attack_against: Some(
                    |state, attacker_choice: &mut Choice, _defender_choice, attacking_side| {
                        if attacker_choice.move_type == PokemonType::Bug
                        || attacker_choice.move_type == PokemonType::Dark
                        || attacker_choice.move_type == PokemonType::Ghost {
                            attacker_choice.add_or_create_secondaries(
                                Secondary {
                                    chance: 100.0,
                                    target: MoveTarget::Opponent,
                                    effect: Effect::Boost(StatBoosts {
                                        attack: 0,
                                        defense: 0,
                                        special_attack: 0,
                                        special_defense: 0,
                                        speed: 1,
                                        accuracy: 0,
                                    }),
                                }
                            );
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "waterbubble".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "sandforce".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        if state.weather_is_active(&Weather::Sand)
                            && (attacking_choice.move_type == PokemonType::Rock
                                || attacking_choice.move_type == PokemonType::Ground
                                || attacking_choice.move_type == PokemonType::Steel)
                        {
                            attacking_choice.base_power *= 1.3;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "toxicboost".to_string(),
            Ability {
                modify_attack_being_used: Some(
                    |state, attacking_choice, defender_choice, attacking_side| {
                        let active_pkmn = state.get_side_immutable(attacking_side).get_active_immutable();
                        if active_pkmn.status == PokemonStatus::Poison
                        || active_pkmn.status == PokemonStatus::Toxic {
                            attacking_choice.base_power *= 1.5;
                        }
                    },
                ),
                ..Default::default()
            },
        );
        abilities.insert(
            "persistent".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "chlorophyll".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities.insert(
            "simple".to_string(),
            Ability {
                ..Default::default()
            },
        );
        abilities
    };
}

impl Default for Ability {
    fn default() -> Ability {
        return Ability {
            modify_attack_being_used: None,
            modify_attack_against: None,
            before_move: None,
            after_damage_hit: None,
            on_switch_out: None,
            on_switch_in: None,
            end_of_turn: None,
        };
    }
}
