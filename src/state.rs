use serde::{Deserialize, Serialize};

use crate::abilities::Abilities;
use crate::choices::{Choice, Choices, MoveCategory, MOVES};
use crate::define_enum_with_from_str;
use crate::instruction::{
    BoostInstruction, ChangeSideConditionInstruction, ChangeStatInstruction, ChangeType,
    ChangeVolatileStatusDurationInstruction, EnableMoveInstruction, Instruction,
    RemoveVolatileStatusInstruction, StateInstructions,
};
use crate::items::Items;
use crate::pokemon::PokemonName;
use core::panic;
use std::collections::HashSet;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

fn common_pkmn_stat_calc(stat: u16, ev: u16, level: u16) -> u16 {
    // 31 IV always used
    ((2 * stat + 31 + (ev / 4)) * level) / 100
}

fn multiply_boost(boost_num: i8, stat_value: i16) -> i16 {
    match boost_num {
        -6 => stat_value * 2 / 8,
        -5 => stat_value * 2 / 7,
        -4 => stat_value * 2 / 6,
        -3 => stat_value * 2 / 5,
        -2 => stat_value * 2 / 4,
        -1 => stat_value * 2 / 3,
        0 => stat_value,
        1 => stat_value * 3 / 2,
        2 => stat_value * 4 / 2,
        3 => stat_value * 5 / 2,
        4 => stat_value * 6 / 2,
        5 => stat_value * 7 / 2,
        6 => stat_value * 8 / 2,
        _ => panic!("Invalid boost number"),
    }
}

#[derive(Debug, Clone)]
pub struct DamageDealt {
    pub damage: i16,
    pub move_category: MoveCategory,
    pub hit_substitute: bool,
}

impl Default for DamageDealt {
    fn default() -> DamageDealt {
        DamageDealt {
            damage: 0,
            move_category: MoveCategory::Physical,
            hit_substitute: false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum LastUsedMove {
    Move(PokemonMoveIndex),
    Switch(PokemonIndex),
    None,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, Serialize, Deserialize)]
pub enum MoveChoice {
    MoveTera(PokemonMoveIndex),
    Move(PokemonMoveIndex),
    Switch(PokemonIndex),
    None,
}

impl MoveChoice {
    pub fn to_string(&self, side: &Side) -> String {
        match self {
            MoveChoice::MoveTera(index) => {
                format!("{}-tera", side.get_active_immutable().moves[index].id).to_lowercase()
            }
            MoveChoice::Move(index) => {
                format!("{}", side.get_active_immutable().moves[index].id).to_lowercase()
            }
            MoveChoice::Switch(index) => {
                format!("switch {}", side.pokemon[*index].id).to_lowercase()
            }
            MoveChoice::None => "No Move".to_string(),
        }
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone)]
    SideMovesFirst {
        SideOne,
        SideTwo,
        SpeedTie
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Clone)]
    PokemonNature {
        HARDY,
        LONELY,
        ADAMANT,
        NAUGHTY,
        BRAVE,
        BOLD,
        DOCILE,
        IMPISH,
        LAX,
        RELAXED,
        MODEST,
        MILD,
        BASHFUL,
        RASH,
        QUIET,
        CALM,
        GENTLE,
        CAREFUL,
        QUIRKY,
        SASSY,
        TIMID,
        HASTY,
        JOLLY,
        NAIVE,
        SERIOUS
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone)]
    PokemonStatus {
        NONE,
        BURN,
        SLEEP,
        FREEZE,
        PARALYZE,
        POISON,
        TOXIC,
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
    PokemonVolatileStatus {
        NONE,
        AQUARING,
        ATTRACT,
        AUTOTOMIZE,
        BANEFULBUNKER,
        BIDE,
        BOUNCE,
        BURNINGBULWARK,
        CHARGE,
        CONFUSION,
        CURSE,
        DEFENSECURL,
        DESTINYBOND,
        DIG,
        DISABLE,
        DIVE,
        ELECTRIFY,
        ELECTROSHOT,
        EMBARGO,
        ENCORE,
        ENDURE,
        FLASHFIRE,
        FLINCH,
        FLY,
        FOCUSENERGY,
        FOLLOWME,
        FORESIGHT,
        FREEZESHOCK,
        GASTROACID,
        GEOMANCY,
        GLAIVERUSH,
        GRUDGE,
        HEALBLOCK,
        HELPINGHAND,
        ICEBURN,
        IMPRISON,
        INGRAIN,
        KINGSSHIELD,
        LASERFOCUS,
        LEECHSEED,
        LIGHTSCREEN,
        LOCKEDMOVE,
        MAGICCOAT,
        MAGNETRISE,
        MAXGUARD,
        METEORBEAM,
        MINIMIZE,
        MIRACLEEYE,
        MUSTRECHARGE,
        NIGHTMARE,
        NORETREAT,
        OCTOLOCK,
        PARTIALLYTRAPPED,
        PERISH4,
        PERISH3,
        PERISH2,
        PERISH1,
        PHANTOMFORCE,
        POWDER,
        POWERSHIFT,
        POWERTRICK,
        PROTECT,
        PROTOSYNTHESISATK,
        PROTOSYNTHESISDEF,
        PROTOSYNTHESISSPA,
        PROTOSYNTHESISSPD,
        PROTOSYNTHESISSPE,
        QUARKDRIVEATK,
        QUARKDRIVEDEF,
        QUARKDRIVESPA,
        QUARKDRIVESPD,
        QUARKDRIVESPE,
        RAGE,
        RAGEPOWDER,
        RAZORWIND,
        REFLECT,
        ROOST,
        SALTCURE,
        SHADOWFORCE,
        SKULLBASH,
        SKYATTACK,
        SKYDROP,
        SILKTRAP,
        SLOWSTART,
        SMACKDOWN,
        SNATCH,
        SOLARBEAM,
        SOLARBLADE,
        SPARKLINGARIA,
        SPIKYSHIELD,
        SPOTLIGHT,
        STOCKPILE,
        SUBSTITUTE,
        SYRUPBOMB,
        TARSHOT,
        TAUNT,
        TELEKINESIS,
        THROATCHOP,
        TRUANT,
        TORMENT,
        TYPECHANGE,
        UNBURDEN,
        UPROAR,
        YAWN,
        YAWNSLEEPTHISTURN,
    },
    default = NONE
}

#[derive(Debug, Eq, PartialEq, Hash, Copy, Clone)]
pub enum PokemonSideCondition {
    AuroraVeil,
    CraftyShield,
    HealingWish,
    LightScreen,
    LuckyChant,
    LunarDance,
    MatBlock,
    Mist,
    Protect,
    QuickGuard,
    Reflect,
    Safeguard,
    Spikes,
    Stealthrock,
    StickyWeb,
    Tailwind,
    ToxicCount,
    ToxicSpikes,
    WideGuard,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SideReference {
    SideOne,
    SideTwo,
}

impl SideReference {
    pub fn get_other_side(&self) -> SideReference {
        match self {
            SideReference::SideOne => SideReference::SideTwo,
            SideReference::SideTwo => SideReference::SideOne,
        }
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone, Eq, Hash, PartialOrd, Ord)]
    Weather {
        NONE,
        SUN,
        RAIN,
        SAND,
        HAIL,
        SNOW,
        HARSHSUN,
        HEAVYRAIN,
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StateWeather {
    pub weather_type: Weather,
    pub turns_remaining: i8,
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone, Eq, PartialOrd, Ord)]
    Terrain {
        NONE,
        ELECTRICTERRAIN,
        PSYCHICTERRAIN,
        MISTYTERRAIN,
        GRASSYTERRAIN,
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct StateTerrain {
    pub terrain_type: Terrain,
    pub turns_remaining: i8,
}

#[derive(Debug, PartialEq, Clone)]
pub struct StateTrickRoom {
    pub active: bool,
    pub turns_remaining: i8,
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    PokemonType {
        NORMAL,
        FIRE,
        WATER,
        ELECTRIC,
        GRASS,
        ICE,
        FIGHTING,
        POISON,
        GROUND,
        FLYING,
        PSYCHIC,
        BUG,
        ROCK,
        GHOST,
        DRAGON,
        DARK,
        STEEL,
        FAIRY,
        STELLAR,
        TYPELESS,
    },
    default = TYPELESS
}

#[derive(Debug, PartialEq, Clone, Copy, Eq, Hash)]
pub enum PokemonBoostableStat {
    Attack,
    Defense,
    SpecialAttack,
    SpecialDefense,
    Speed,
    Evasion,
    Accuracy,
}

#[derive(Debug, Clone)]
pub struct VolatileStatusDurations {
    pub confusion: i8,
    pub encore: i8,
    pub lockedmove: i8,
}

impl Default for VolatileStatusDurations {
    fn default() -> VolatileStatusDurations {
        VolatileStatusDurations {
            confusion: 0,
            encore: 0,
            lockedmove: 0,
        }
    }
}

impl VolatileStatusDurations {
    pub fn serialize(&self) -> String {
        format!("{};{};{}", self.confusion, self.encore, self.lockedmove)
    }
    pub fn deserialize(serialized: &str) -> VolatileStatusDurations {
        let split: Vec<&str> = serialized.split(";").collect();
        VolatileStatusDurations {
            confusion: split[0].parse::<i8>().unwrap(),
            encore: split[1].parse::<i8>().unwrap(),
            lockedmove: split[2].parse::<i8>().unwrap(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct SideConditions {
    pub aurora_veil: i8,
    pub crafty_shield: i8,
    pub healing_wish: i8,
    pub light_screen: i8,
    pub lucky_chant: i8,
    pub lunar_dance: i8,
    pub mat_block: i8,
    pub mist: i8,
    pub protect: i8,
    pub quick_guard: i8,
    pub reflect: i8,
    pub safeguard: i8,
    pub spikes: i8,
    pub stealth_rock: i8,
    pub sticky_web: i8,
    pub tailwind: i8,
    pub toxic_count: i8,
    pub toxic_spikes: i8,
    pub wide_guard: i8,
}

impl Default for SideConditions {
    fn default() -> SideConditions {
        SideConditions {
            aurora_veil: 0,
            crafty_shield: 0,
            healing_wish: 0,
            light_screen: 0,
            lucky_chant: 0,
            lunar_dance: 0,
            mat_block: 0,
            mist: 0,
            protect: 0,
            quick_guard: 0,
            reflect: 0,
            safeguard: 0,
            spikes: 0,
            stealth_rock: 0,
            sticky_web: 0,
            tailwind: 0,
            toxic_count: 0,
            toxic_spikes: 0,
            wide_guard: 0,
        }
    }
}

#[derive(Debug, Copy, PartialEq, Clone, Eq, Hash, Serialize, Deserialize)]
pub enum PokemonMoveIndex {
    M0,
    M1,
    M2,
    M3,
}

#[derive(Debug, Clone)]
pub struct PokemonMoves {
    pub m0: Move,
    pub m1: Move,
    pub m2: Move,
    pub m3: Move,
}

impl Index<&PokemonMoveIndex> for PokemonMoves {
    type Output = Move;

    fn index(&self, index: &PokemonMoveIndex) -> &Self::Output {
        match index {
            PokemonMoveIndex::M0 => &self.m0,
            PokemonMoveIndex::M1 => &self.m1,
            PokemonMoveIndex::M2 => &self.m2,
            PokemonMoveIndex::M3 => &self.m3,
        }
    }
}

impl IndexMut<&PokemonMoveIndex> for PokemonMoves {
    fn index_mut(&mut self, index: &PokemonMoveIndex) -> &mut Self::Output {
        match index {
            PokemonMoveIndex::M0 => &mut self.m0,
            PokemonMoveIndex::M1 => &mut self.m1,
            PokemonMoveIndex::M2 => &mut self.m2,
            PokemonMoveIndex::M3 => &mut self.m3,
        }
    }
}

pub struct PokemonMoveIterator<'a> {
    pub pokemon_move: &'a PokemonMoves,
    pub pokemon_move_index: PokemonMoveIndex,
    pub index: usize,
}

impl<'a> Iterator for PokemonMoveIterator<'a> {
    type Item = &'a Move;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index {
            0 => {
                self.index += 1;
                self.pokemon_move_index = PokemonMoveIndex::M0;
                Some(&self.pokemon_move.m0)
            }
            1 => {
                self.index += 1;
                self.pokemon_move_index = PokemonMoveIndex::M1;
                Some(&self.pokemon_move.m1)
            }
            2 => {
                self.index += 1;
                self.pokemon_move_index = PokemonMoveIndex::M2;
                Some(&self.pokemon_move.m2)
            }
            3 => {
                self.index += 1;
                self.pokemon_move_index = PokemonMoveIndex::M3;
                Some(&self.pokemon_move.m3)
            }
            _ => None,
        }
    }
}

impl<'a> IntoIterator for &'a PokemonMoves {
    type Item = &'a Move;
    type IntoIter = PokemonMoveIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PokemonMoveIterator {
            pokemon_move: &self,
            pokemon_move_index: PokemonMoveIndex::M0,
            index: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Move {
    pub id: Choices,
    pub disabled: bool,
    pub pp: i8,
    pub choice: Choice,
}

impl Default for Move {
    fn default() -> Move {
        Move {
            id: Choices::NONE,
            disabled: false,
            pp: 32,
            choice: Choice::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pokemon {
    pub id: PokemonName,
    pub level: i8,
    pub types: (PokemonType, PokemonType),
    pub base_types: (PokemonType, PokemonType),
    pub hp: i16,
    pub maxhp: i16,
    pub ability: Abilities,
    pub base_ability: Abilities,
    pub item: Items,
    pub nature: PokemonNature,
    pub evs: (u8, u8, u8, u8, u8, u8),
    pub attack: i16,
    pub defense: i16,
    pub special_attack: i16,
    pub special_defense: i16,
    pub speed: i16,
    pub status: PokemonStatus,
    pub rest_turns: i8,
    pub sleep_turns: i8,
    pub weight_kg: f32,
    pub terastallized: bool,
    pub tera_type: PokemonType,
    pub moves: PokemonMoves,
}

impl Pokemon {
    pub fn recalculate_stats(
        &mut self,
        side_ref: &SideReference,
        instructions: &mut StateInstructions,
    ) {
        // recalculate stats from base-stats and push any changes made to the StateInstructions
        let stats = self.calculate_stats_from_base_stats();
        if stats.1 != self.attack {
            let ins = Instruction::ChangeAttack(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.1 - self.attack,
            });
            self.attack = stats.1;
            instructions.instruction_list.push(ins);
        }
        if stats.2 != self.defense {
            let ins = Instruction::ChangeDefense(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.2 - self.defense,
            });
            self.defense = stats.2;
            instructions.instruction_list.push(ins);
        }
        if stats.3 != self.special_attack {
            let ins = Instruction::ChangeSpecialAttack(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.3 - self.special_attack,
            });
            self.special_attack = stats.3;
            instructions.instruction_list.push(ins);
        }
        if stats.4 != self.special_defense {
            let ins = Instruction::ChangeSpecialDefense(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.4 - self.special_defense,
            });
            self.special_defense = stats.4;
            instructions.instruction_list.push(ins);
        }
        if stats.5 != self.speed {
            let ins = Instruction::ChangeSpeed(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.5 - self.speed,
            });
            self.speed = stats.5;
            instructions.instruction_list.push(ins);
        }
    }
    pub fn calculate_stats_from_base_stats(&self) -> (i16, i16, i16, i16, i16, i16) {
        let base_stats = self.id.base_stats();
        (
            (common_pkmn_stat_calc(base_stats.0 as u16, self.evs.0 as u16, self.level as u16)
                + self.level as u16
                + 10) as i16,
            (common_pkmn_stat_calc(base_stats.1 as u16, self.evs.1 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.2 as u16, self.evs.2 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.3 as u16, self.evs.3 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.4 as u16, self.evs.4 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.5 as u16, self.evs.5 as u16, self.level as u16) + 5)
                as i16,
        )
    }
    pub fn get_stat_from_boostable_stat(&self, stat: PokemonBoostableStat) -> i16 {
        match stat {
            PokemonBoostableStat::Attack => self.attack,
            PokemonBoostableStat::Defense => self.defense,
            PokemonBoostableStat::SpecialAttack => self.special_attack,
            PokemonBoostableStat::SpecialDefense => self.special_defense,
            PokemonBoostableStat::Speed => self.speed,
            _ => panic!("Not implemented"),
        }
    }
    pub fn get_sleep_talk_choices(&self) -> Vec<Choice> {
        let mut vec = Vec::with_capacity(4);
        for p in self.moves.into_iter() {
            if p.id != Choices::SLEEPTALK && p.id != Choices::NONE {
                vec.push(p.choice.clone());
            }
        }
        vec
    }
    pub fn replace_move(&mut self, move_index: PokemonMoveIndex, new_move_name: Choices) {
        self.moves[&move_index].choice = MOVES.get(&new_move_name).unwrap().to_owned();
        self.moves[&move_index].id = new_move_name;
    }

    pub fn add_available_moves(
        &self,
        vec: &mut Vec<MoveChoice>,
        last_used_move: &LastUsedMove,
        encored: bool,
        can_tera: bool,
    ) {
        let mut iter = self.moves.into_iter();
        while let Some(p) = iter.next() {
            if !p.disabled && p.pp > 0 {
                match last_used_move {
                    LastUsedMove::Move(last_used_move) => {
                        if encored && last_used_move != &iter.pokemon_move_index {
                            continue;
                        } else if (self.moves[last_used_move].id == Choices::BLOODMOON
                            || self.moves[last_used_move].id == Choices::GIGATONHAMMER)
                            && &iter.pokemon_move_index == last_used_move
                        {
                            continue;
                        }
                    }
                    _ => {
                        // there are some situations where you switched out and got encored into
                        // a move from a different pokemon because you also have that move.
                        // just assume nothing is locked in this case
                    }
                }
                if self.item == Items::ASSAULTVEST
                    && self.moves[&iter.pokemon_move_index].choice.category == MoveCategory::Status
                {
                    continue;
                }
                vec.push(MoveChoice::Move(iter.pokemon_move_index));
                if can_tera {
                    vec.push(MoveChoice::MoveTera(iter.pokemon_move_index));
                }
            }
        }
    }

    pub fn add_move_from_choice(&self, vec: &mut Vec<MoveChoice>, choice: Choices) {
        let mut iter = self.moves.into_iter();
        while let Some(p) = iter.next() {
            if p.id == choice {
                vec.push(MoveChoice::Move(iter.pokemon_move_index));
            }
        }
    }

    #[cfg(feature = "terastallization")]
    pub fn has_type(&self, pkmn_type: &PokemonType) -> bool {
        if self.terastallized {
            pkmn_type == &self.tera_type
        } else {
            pkmn_type == &self.types.0 || pkmn_type == &self.types.1
        }
    }

    #[cfg(not(feature = "terastallization"))]
    pub fn has_type(&self, pkmn_type: &PokemonType) -> bool {
        pkmn_type == &self.types.0 || pkmn_type == &self.types.1
    }

    pub fn item_is_permanent(&self) -> bool {
        match self.item {
            Items::LUSTROUSGLOBE => self.id == PokemonName::PALKIAORIGIN,
            Items::GRISEOUSCORE => self.id == PokemonName::GIRATINAORIGIN,
            Items::ADAMANTCRYSTAL => self.id == PokemonName::DIALGAORIGIN,
            Items::RUSTEDSWORD => {
                self.id == PokemonName::ZACIANCROWNED || self.id == PokemonName::ZACIAN
            }
            Items::RUSTEDSHIELD => {
                self.id == PokemonName::ZAMAZENTACROWNED || self.id == PokemonName::ZAMAZENTA
            }
            Items::SPLASHPLATE => self.id == PokemonName::ARCEUSWATER,
            Items::TOXICPLATE => self.id == PokemonName::ARCEUSPOISON,
            Items::EARTHPLATE => self.id == PokemonName::ARCEUSGROUND,
            Items::STONEPLATE => self.id == PokemonName::ARCEUSROCK,
            Items::INSECTPLATE => self.id == PokemonName::ARCEUSBUG,
            Items::SPOOKYPLATE => self.id == PokemonName::ARCEUSGHOST,
            Items::IRONPLATE => self.id == PokemonName::ARCEUSSTEEL,
            Items::FLAMEPLATE => self.id == PokemonName::ARCEUSFIRE,
            Items::MEADOWPLATE => self.id == PokemonName::ARCEUSGRASS,
            Items::ZAPPLATE => self.id == PokemonName::ARCEUSELECTRIC,
            Items::MINDPLATE => self.id == PokemonName::ARCEUSPSYCHIC,
            Items::ICICLEPLATE => self.id == PokemonName::ARCEUSICE,
            Items::DRACOPLATE => self.id == PokemonName::ARCEUSDRAGON,
            Items::DREADPLATE => self.id == PokemonName::ARCEUSDARK,
            Items::FISTPLATE => self.id == PokemonName::ARCEUSFIGHTING,
            Items::BLANKPLATE => self.id == PokemonName::ARCEUS,
            Items::SKYPLATE => self.id == PokemonName::ARCEUSFLYING,
            Items::PIXIEPLATE => self.id == PokemonName::ARCEUSFAIRY,
            Items::BUGMEMORY => self.id == PokemonName::SILVALLYBUG,
            Items::FIGHTINGMEMORY => self.id == PokemonName::SILVALLYFIGHTING,
            Items::GHOSTMEMORY => self.id == PokemonName::SILVALLYGHOST,
            Items::PSYCHICMEMORY => self.id == PokemonName::SILVALLYPSYCHIC,
            Items::FLYINGMEMORY => self.id == PokemonName::SILVALLYFLYING,
            Items::STEELMEMORY => self.id == PokemonName::SILVALLYSTEEL,
            Items::ICEMEMORY => self.id == PokemonName::SILVALLYICE,
            Items::POISONMEMORY => self.id == PokemonName::SILVALLYPOISON,
            Items::FIREMEMORY => self.id == PokemonName::SILVALLYFIRE,
            Items::DRAGONMEMORY => self.id == PokemonName::SILVALLYDRAGON,
            Items::GROUNDMEMORY => self.id == PokemonName::SILVALLYGROUND,
            Items::WATERMEMORY => self.id == PokemonName::SILVALLYWATER,
            Items::DARKMEMORY => self.id == PokemonName::SILVALLYDARK,
            Items::ROCKMEMORY => self.id == PokemonName::SILVALLYROCK,
            Items::GRASSMEMORY => self.id == PokemonName::SILVALLYGRASS,
            Items::FAIRYMEMORY => self.id == PokemonName::SILVALLYFAIRY,
            Items::ELECTRICMEMORY => self.id == PokemonName::SILVALLYELECTRIC,
            Items::CORNERSTONEMASK => {
                self.id == PokemonName::OGERPONCORNERSTONE
                    || self.id == PokemonName::OGERPONCORNERSTONETERA
            }
            Items::HEARTHFLAMEMASK => {
                self.id == PokemonName::OGERPONHEARTHFLAME
                    || self.id == PokemonName::OGERPONHEARTHFLAMETERA
            }
            Items::WELLSPRINGMASK => {
                self.id == PokemonName::OGERPONWELLSPRING
                    || self.id == PokemonName::OGERPONWELLSPRINGTERA
            }
            _ => false,
        }
    }

    pub fn item_can_be_removed(&self) -> bool {
        if self.ability == Abilities::STICKYHOLD {
            return false;
        }
        !self.item_is_permanent()
    }

    pub fn is_grounded(&self) -> bool {
        if self.item == Items::IRONBALL {
            return true;
        }
        if self.has_type(&PokemonType::FLYING)
            || self.ability == Abilities::LEVITATE
            || self.item == Items::AIRBALLOON
        {
            return false;
        }
        true
    }

    pub fn volatile_status_can_be_applied(
        &self,
        volatile_status: &PokemonVolatileStatus,
        active_volatiles: &HashSet<PokemonVolatileStatus>,
        first_move: bool,
    ) -> bool {
        if active_volatiles.contains(volatile_status) || self.hp == 0 {
            return false;
        }
        match volatile_status {
            PokemonVolatileStatus::LEECHSEED => {
                if self.has_type(&PokemonType::GRASS)
                    || active_volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE)
                {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::CONFUSION => {
                if active_volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE) {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::SUBSTITUTE => self.hp > self.maxhp / 4,
            PokemonVolatileStatus::FLINCH => {
                if !first_move || [Abilities::INNERFOCUS].contains(&self.ability) {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::PROTECT => first_move,
            PokemonVolatileStatus::TAUNT
            | PokemonVolatileStatus::TORMENT
            | PokemonVolatileStatus::ENCORE
            | PokemonVolatileStatus::DISABLE
            | PokemonVolatileStatus::HEALBLOCK
            | PokemonVolatileStatus::ATTRACT => self.ability != Abilities::AROMAVEIL,
            PokemonVolatileStatus::YAWN => {
                // immunity to yawn via sleep immunity is handled in `get_instructions_from_volatile_statuses`
                !active_volatiles.contains(&PokemonVolatileStatus::YAWNSLEEPTHISTURN)
            }
            _ => true,
        }
    }

    pub fn immune_to_stats_lowered_by_opponent(
        &self,
        stat: &PokemonBoostableStat,
        volatiles: &HashSet<PokemonVolatileStatus>,
    ) -> bool {
        if [
            Abilities::CLEARBODY,
            Abilities::WHITESMOKE,
            Abilities::FULLMETALBODY,
        ]
        .contains(&self.ability)
            || ([Items::CLEARAMULET].contains(&self.item))
        {
            return true;
        }

        if volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE) {
            return true;
        }

        if stat == &PokemonBoostableStat::Attack && self.ability == Abilities::HYPERCUTTER {
            return true;
        } else if stat == &PokemonBoostableStat::Accuracy && self.ability == Abilities::KEENEYE {
            return true;
        }

        false
    }
}

impl Default for Pokemon {
    fn default() -> Pokemon {
        Pokemon {
            id: PokemonName::NONE,
            level: 100,
            types: (PokemonType::NORMAL, PokemonType::TYPELESS),
            base_types: (PokemonType::NORMAL, PokemonType::TYPELESS),
            hp: 0,
            maxhp: 100,
            ability: Abilities::NONE,
            base_ability: Abilities::NONE,
            item: Items::NONE,
            nature: PokemonNature::SERIOUS,
            evs: (85, 85, 85, 85, 85, 85),
            attack: 100,
            defense: 100,
            special_attack: 100,
            special_defense: 100,
            speed: 100,
            status: PokemonStatus::NONE,
            rest_turns: 0,
            sleep_turns: 0,
            weight_kg: 1.0,
            terastallized: false,
            tera_type: PokemonType::NORMAL,
            moves: PokemonMoves {
                m0: Default::default(),
                m1: Default::default(),
                m2: Default::default(),
                m3: Default::default(),
            },
        }
    }
}

#[derive(Debug, Copy, PartialEq, Clone, Eq, Hash, Serialize, Deserialize)]
pub enum PokemonIndex {
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
}

pub fn pokemon_index_iter() -> PokemonIndexIterator {
    PokemonIndexIterator { index: 0 }
}

pub struct PokemonIndexIterator {
    index: usize,
}

impl Iterator for PokemonIndexIterator {
    type Item = PokemonIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index {
            0 => {
                self.index += 1;
                Some(PokemonIndex::P0)
            }
            1 => {
                self.index += 1;
                Some(PokemonIndex::P1)
            }
            2 => {
                self.index += 1;
                Some(PokemonIndex::P2)
            }
            3 => {
                self.index += 1;
                Some(PokemonIndex::P3)
            }
            4 => {
                self.index += 1;
                Some(PokemonIndex::P4)
            }
            5 => {
                self.index += 1;
                Some(PokemonIndex::P5)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SidePokemon {
    pub p0: Pokemon,
    pub p1: Pokemon,
    pub p2: Pokemon,
    pub p3: Pokemon,
    pub p4: Pokemon,
    pub p5: Pokemon,
}

impl<'a> IntoIterator for &'a SidePokemon {
    type Item = &'a Pokemon;
    type IntoIter = SidePokemonIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        SidePokemonIterator {
            side_pokemon: &self,
            pokemon_index: PokemonIndex::P0,
            index: 0,
        }
    }
}

pub struct SidePokemonIterator<'a> {
    pub side_pokemon: &'a SidePokemon,
    pub pokemon_index: PokemonIndex,
    pub index: usize,
}

impl<'a> Iterator for SidePokemonIterator<'a> {
    type Item = &'a Pokemon;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index {
            0 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P0;
                Some(&self.side_pokemon.p0)
            }
            1 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P1;
                Some(&self.side_pokemon.p1)
            }
            2 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P2;
                Some(&self.side_pokemon.p2)
            }
            3 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P3;
                Some(&self.side_pokemon.p3)
            }
            4 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P4;
                Some(&self.side_pokemon.p4)
            }
            5 => {
                self.index += 1;
                self.pokemon_index = PokemonIndex::P5;
                Some(&self.side_pokemon.p5)
            }
            _ => None,
        }
    }
}

impl Index<PokemonIndex> for SidePokemon {
    type Output = Pokemon;

    fn index(&self, index: PokemonIndex) -> &Self::Output {
        match index {
            PokemonIndex::P0 => &self.p0,
            PokemonIndex::P1 => &self.p1,
            PokemonIndex::P2 => &self.p2,
            PokemonIndex::P3 => &self.p3,
            PokemonIndex::P4 => &self.p4,
            PokemonIndex::P5 => &self.p5,
        }
    }
}

impl Index<&PokemonIndex> for SidePokemon {
    type Output = Pokemon;

    fn index(&self, index: &PokemonIndex) -> &Self::Output {
        match index {
            PokemonIndex::P0 => &self.p0,
            PokemonIndex::P1 => &self.p1,
            PokemonIndex::P2 => &self.p2,
            PokemonIndex::P3 => &self.p3,
            PokemonIndex::P4 => &self.p4,
            PokemonIndex::P5 => &self.p5,
        }
    }
}

impl IndexMut<PokemonIndex> for SidePokemon {
    fn index_mut(&mut self, index: PokemonIndex) -> &mut Self::Output {
        match index {
            PokemonIndex::P0 => &mut self.p0,
            PokemonIndex::P1 => &mut self.p1,
            PokemonIndex::P2 => &mut self.p2,
            PokemonIndex::P3 => &mut self.p3,
            PokemonIndex::P4 => &mut self.p4,
            PokemonIndex::P5 => &mut self.p5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Side {
    pub active_index: PokemonIndex,
    pub baton_passing: bool,
    pub pokemon: SidePokemon,
    pub side_conditions: SideConditions,
    pub volatile_status_durations: VolatileStatusDurations,
    pub wish: (i8, i16),
    pub future_sight: (i8, PokemonIndex),
    pub force_switch: bool,
    pub force_trapped: bool,
    pub slow_uturn_move: bool,
    pub volatile_statuses: HashSet<PokemonVolatileStatus>,
    pub substitute_health: i16,
    pub attack_boost: i8,
    pub defense_boost: i8,
    pub special_attack_boost: i8,
    pub special_defense_boost: i8,
    pub speed_boost: i8,
    pub accuracy_boost: i8,
    pub evasion_boost: i8,
    pub last_used_move: LastUsedMove,
    pub damage_dealt: DamageDealt,
    pub switch_out_move_second_saved_move: Choices,
}

impl Side {
    pub fn calculate_highest_stat(&self) -> PokemonBoostableStat {
        let mut highest_stat = PokemonBoostableStat::Attack;
        let mut highest_stat_value = self.calculate_boosted_stat(PokemonBoostableStat::Attack);
        for stat in [
            PokemonBoostableStat::Defense,
            PokemonBoostableStat::SpecialAttack,
            PokemonBoostableStat::SpecialDefense,
            PokemonBoostableStat::Speed,
        ] {
            let stat_value = self.calculate_boosted_stat(stat);
            if stat_value > highest_stat_value {
                highest_stat = stat;
                highest_stat_value = stat_value;
            }
        }
        highest_stat
    }
    pub fn get_boost_from_boost_enum(&self, boost_enum: &PokemonBoostableStat) -> i8 {
        match boost_enum {
            PokemonBoostableStat::Attack => self.attack_boost,
            PokemonBoostableStat::Defense => self.defense_boost,
            PokemonBoostableStat::SpecialAttack => self.special_attack_boost,
            PokemonBoostableStat::SpecialDefense => self.special_defense_boost,
            PokemonBoostableStat::Speed => self.speed_boost,
            PokemonBoostableStat::Evasion => self.evasion_boost,
            PokemonBoostableStat::Accuracy => self.accuracy_boost,
        }
    }

    pub fn calculate_boosted_stat(&self, stat: PokemonBoostableStat) -> i16 {
        /*
        In Gen4, simple doubles the effective boost, without it visually being doubled
        It will not boost beyond an effective value of 6 though.
        */
        let active = self.get_active_immutable();
        match stat {
            PokemonBoostableStat::Attack => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.attack_boost * 2).min(6).max(-6)
                } else {
                    self.attack_boost
                };

                #[cfg(not(feature = "gen4"))]
                let boost = self.attack_boost;

                multiply_boost(boost, active.attack)
            }
            PokemonBoostableStat::Defense => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.defense_boost * 2).min(6).max(-6)
                } else {
                    self.defense_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.defense_boost;

                multiply_boost(boost, active.defense)
            }
            PokemonBoostableStat::SpecialAttack => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.special_attack_boost * 2).min(6).max(-6)
                } else {
                    self.special_attack_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.special_attack_boost;

                multiply_boost(boost, active.special_attack)
            }
            PokemonBoostableStat::SpecialDefense => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.special_defense_boost * 2).min(6).max(-6)
                } else {
                    self.special_defense_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.special_defense_boost;

                multiply_boost(boost, active.special_defense)
            }
            PokemonBoostableStat::Speed => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.speed_boost * 2).min(6).max(-6)
                } else {
                    self.speed_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.speed_boost;

                multiply_boost(boost, active.speed)
            }
            _ => {
                panic!("Not implemented")
            }
        }
    }

    pub fn has_alive_non_rested_sleeping_pkmn(&self) -> bool {
        for p in self.pokemon.into_iter() {
            if p.status == PokemonStatus::SLEEP && p.hp > 0 && p.rest_turns == 0 {
                return true;
            }
        }
        false
    }

    #[cfg(not(feature = "terastallization"))]
    pub fn can_use_tera(&self) -> bool {
        false
    }

    #[cfg(feature = "terastallization")]
    pub fn can_use_tera(&self) -> bool {
        for p in self.pokemon.into_iter() {
            if p.terastallized {
                return false;
            }
        }
        true
    }

    fn toggle_force_switch(&mut self) {
        self.force_switch = !self.force_switch;
    }

    pub fn add_switches(&self, vec: &mut Vec<MoveChoice>) {
        let mut iter = self.pokemon.into_iter();
        while let Some(p) = iter.next() {
            if p.hp > 0 && iter.pokemon_index != self.active_index {
                vec.push(MoveChoice::Switch(iter.pokemon_index));
            }
        }
        if vec.len() == 0 {
            vec.push(MoveChoice::None);
        }
    }

    pub fn trapped(&self, opponent_active: &Pokemon) -> bool {
        let active_pkmn = self.get_active_immutable();
        if self
            .volatile_statuses
            .contains(&PokemonVolatileStatus::LOCKEDMOVE)
        {
            return true;
        }
        if active_pkmn.item == Items::SHEDSHELL || active_pkmn.has_type(&PokemonType::GHOST) {
            return false;
        } else if self
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PARTIALLYTRAPPED)
        {
            return true;
        } else if opponent_active.ability == Abilities::SHADOWTAG {
            return true;
        } else if opponent_active.ability == Abilities::ARENATRAP && active_pkmn.is_grounded() {
            return true;
        } else if opponent_active.ability == Abilities::MAGNETPULL
            && active_pkmn.has_type(&PokemonType::STEEL)
        {
            return true;
        }
        false
    }

    pub fn get_active(&mut self) -> &mut Pokemon {
        &mut self.pokemon[self.active_index]
    }

    pub fn get_active_immutable(&self) -> &Pokemon {
        &self.pokemon[self.active_index]
    }

    pub fn num_fainted_pkmn(&self) -> i8 {
        let mut count = 0;
        for p in self.pokemon.into_iter() {
            if p.hp == 0 && p.level != 1 {
                count += 1;
            }
        }
        count
    }

    pub fn visible_alive_pkmn(&self) -> i8 {
        let mut count = 0;
        for p in self.pokemon.into_iter() {
            if p.hp > 0 {
                count += 1;
            }
        }
        count
    }

    pub fn get_alive_pkmn_indices(&self) -> Vec<PokemonIndex> {
        let mut vec = Vec::with_capacity(6);
        let mut iter = self.pokemon.into_iter();

        while let Some(p) = iter.next() {
            if p.hp > 0 && iter.pokemon_index != self.active_index {
                vec.push(iter.pokemon_index.clone());
            }
        }

        vec
    }

    pub fn get_side_condition(&self, side_condition: PokemonSideCondition) -> i8 {
        match side_condition {
            PokemonSideCondition::AuroraVeil => self.side_conditions.aurora_veil,
            PokemonSideCondition::CraftyShield => self.side_conditions.crafty_shield,
            PokemonSideCondition::HealingWish => self.side_conditions.healing_wish,
            PokemonSideCondition::LightScreen => self.side_conditions.light_screen,
            PokemonSideCondition::LuckyChant => self.side_conditions.lucky_chant,
            PokemonSideCondition::LunarDance => self.side_conditions.lunar_dance,
            PokemonSideCondition::MatBlock => self.side_conditions.mat_block,
            PokemonSideCondition::Mist => self.side_conditions.mist,
            PokemonSideCondition::Protect => self.side_conditions.protect,
            PokemonSideCondition::QuickGuard => self.side_conditions.quick_guard,
            PokemonSideCondition::Reflect => self.side_conditions.reflect,
            PokemonSideCondition::Safeguard => self.side_conditions.safeguard,
            PokemonSideCondition::Spikes => self.side_conditions.spikes,
            PokemonSideCondition::Stealthrock => self.side_conditions.stealth_rock,
            PokemonSideCondition::StickyWeb => self.side_conditions.sticky_web,
            PokemonSideCondition::Tailwind => self.side_conditions.tailwind,
            PokemonSideCondition::ToxicCount => self.side_conditions.toxic_count,
            PokemonSideCondition::ToxicSpikes => self.side_conditions.toxic_spikes,
            PokemonSideCondition::WideGuard => self.side_conditions.wide_guard,
        }
    }

    pub fn update_side_condition(&mut self, side_condition: PokemonSideCondition, amount: i8) {
        match side_condition {
            PokemonSideCondition::AuroraVeil => self.side_conditions.aurora_veil += amount,
            PokemonSideCondition::CraftyShield => self.side_conditions.crafty_shield += amount,
            PokemonSideCondition::HealingWish => self.side_conditions.healing_wish += amount,
            PokemonSideCondition::LightScreen => self.side_conditions.light_screen += amount,
            PokemonSideCondition::LuckyChant => self.side_conditions.lucky_chant += amount,
            PokemonSideCondition::LunarDance => self.side_conditions.lunar_dance += amount,
            PokemonSideCondition::MatBlock => self.side_conditions.mat_block += amount,
            PokemonSideCondition::Mist => self.side_conditions.mist += amount,
            PokemonSideCondition::Protect => self.side_conditions.protect += amount,
            PokemonSideCondition::QuickGuard => self.side_conditions.quick_guard += amount,
            PokemonSideCondition::Reflect => self.side_conditions.reflect += amount,
            PokemonSideCondition::Safeguard => self.side_conditions.safeguard += amount,
            PokemonSideCondition::Spikes => self.side_conditions.spikes += amount,
            PokemonSideCondition::Stealthrock => self.side_conditions.stealth_rock += amount,
            PokemonSideCondition::StickyWeb => self.side_conditions.sticky_web += amount,
            PokemonSideCondition::Tailwind => self.side_conditions.tailwind += amount,
            PokemonSideCondition::ToxicCount => self.side_conditions.toxic_count += amount,
            PokemonSideCondition::ToxicSpikes => self.side_conditions.toxic_spikes += amount,
            PokemonSideCondition::WideGuard => self.side_conditions.wide_guard += amount,
        }
    }
}

impl Default for Side {
    fn default() -> Side {
        Side {
            active_index: PokemonIndex::P0,
            baton_passing: false,
            pokemon: SidePokemon {
                p0: Pokemon {
                    ..Pokemon::default()
                },
                p1: Pokemon {
                    ..Pokemon::default()
                },
                p2: Pokemon {
                    ..Pokemon::default()
                },
                p3: Pokemon {
                    ..Pokemon::default()
                },
                p4: Pokemon {
                    ..Pokemon::default()
                },
                p5: Pokemon {
                    ..Pokemon::default()
                },
            },
            substitute_health: 0,
            attack_boost: 0,
            defense_boost: 0,
            special_attack_boost: 0,
            special_defense_boost: 0,
            speed_boost: 0,
            accuracy_boost: 0,
            side_conditions: SideConditions {
                ..Default::default()
            },
            volatile_status_durations: VolatileStatusDurations::default(),
            volatile_statuses: HashSet::<PokemonVolatileStatus>::new(),
            wish: (0, 0),
            future_sight: (0, PokemonIndex::P0),
            force_switch: false,
            slow_uturn_move: false,
            force_trapped: false,
            last_used_move: LastUsedMove::None,
            damage_dealt: DamageDealt::default(),
            switch_out_move_second_saved_move: Choices::NONE,
            evasion_boost: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    pub side_one: Side,
    pub side_two: Side,
    pub weather: StateWeather,
    pub terrain: StateTerrain,
    pub trick_room: StateTrickRoom,
    pub team_preview: bool,
    pub use_last_used_move: bool,
    pub use_damage_dealt: bool,
}

impl Default for State {
    fn default() -> State {
        let mut s = State {
            side_one: Side::default(),
            side_two: Side::default(),
            weather: StateWeather {
                weather_type: Weather::NONE,
                turns_remaining: -1,
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
            use_damage_dealt: false,
            use_last_used_move: false,
        };

        // many tests rely on the speed of side 2's active pokemon being greater than side_one's
        s.side_two.get_active().speed += 1;
        s
    }
}

impl State {
    pub fn battle_is_over(&self) -> f32 {
        //  0 if battle is not over
        //  1 if side one has won
        // -1 if side two has won
        if self.side_one.pokemon.into_iter().all(|p| p.hp <= 0) {
            return -1.0;
        }
        // level == 1 represents an un-revealed pokemon
        if self
            .side_two
            .pokemon
            .into_iter()
            .all(|p| p.hp <= 0 && p.level != 1)
        {
            return 1.0;
        }
        0.0
    }

    pub fn get_all_options(&self) -> (Vec<MoveChoice>, Vec<MoveChoice>) {
        let mut side_one_options: Vec<MoveChoice> = Vec::with_capacity(9);
        let mut side_two_options: Vec<MoveChoice> = Vec::with_capacity(9);

        let side_one_active = self.side_one.get_active_immutable();
        let side_two_active = self.side_two.get_active_immutable();

        if self.side_one.force_switch {
            self.side_one.add_switches(&mut side_one_options);
            if self.side_two.switch_out_move_second_saved_move == Choices::NONE {
                side_two_options.push(MoveChoice::None);
            } else {
                self.side_two.get_active_immutable().add_move_from_choice(
                    &mut side_two_options,
                    self.side_two.switch_out_move_second_saved_move,
                );
            }
            return (side_one_options, side_two_options);
        }

        if self.side_two.force_switch {
            self.side_two.add_switches(&mut side_two_options);
            if self.side_one.switch_out_move_second_saved_move == Choices::NONE {
                side_one_options.push(MoveChoice::None);
            } else {
                self.side_one.get_active_immutable().add_move_from_choice(
                    &mut side_one_options,
                    self.side_one.switch_out_move_second_saved_move,
                );
            }
            return (side_one_options, side_two_options);
        }

        let side_one_force_switch = self.side_one.get_active_immutable().hp <= 0;
        let side_two_force_switch = self.side_two.get_active_immutable().hp <= 0;

        if side_one_force_switch && side_two_force_switch {
            self.side_one.add_switches(&mut side_one_options);
            self.side_two.add_switches(&mut side_two_options);
            return (side_one_options, side_two_options);
        }
        if side_one_force_switch {
            self.side_one.add_switches(&mut side_one_options);
            side_two_options.push(MoveChoice::None);
            return (side_one_options, side_two_options);
        }
        if side_two_force_switch {
            side_one_options.push(MoveChoice::None);
            self.side_two.add_switches(&mut side_two_options);
            return (side_one_options, side_two_options);
        }

        if self
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE)
        {
            side_one_options.push(MoveChoice::None);
        } else {
            let encored = self
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            self.side_one.get_active_immutable().add_available_moves(
                &mut side_one_options,
                &self.side_one.last_used_move,
                encored,
                self.side_one.can_use_tera(),
            );
            if !self.side_one.trapped(side_two_active) {
                self.side_one.add_switches(&mut side_one_options);
            }
        }

        if self
            .side_two
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE)
        {
            side_two_options.push(MoveChoice::None);
        } else {
            let encored = self
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            self.side_two.get_active_immutable().add_available_moves(
                &mut side_two_options,
                &self.side_two.last_used_move,
                encored,
                self.side_two.can_use_tera(),
            );
            if !self.side_two.trapped(side_one_active) {
                self.side_two.add_switches(&mut side_two_options);
            }
        }

        if side_one_options.len() == 0 {
            side_one_options.push(MoveChoice::None);
        }
        if side_two_options.len() == 0 {
            side_two_options.push(MoveChoice::None);
        }

        (side_one_options, side_two_options)
    }

    pub fn get_side(&mut self, side_ref: &SideReference) -> &mut Side {
        match side_ref {
            SideReference::SideOne => &mut self.side_one,
            SideReference::SideTwo => &mut self.side_two,
        }
    }

    pub fn get_side_immutable(&self, side_ref: &SideReference) -> &Side {
        match side_ref {
            SideReference::SideOne => &self.side_one,
            SideReference::SideTwo => &self.side_two,
        }
    }

    pub fn get_both_sides(&mut self, side_ref: &SideReference) -> (&mut Side, &mut Side) {
        match side_ref {
            SideReference::SideOne => (&mut self.side_one, &mut self.side_two),
            SideReference::SideTwo => (&mut self.side_two, &mut self.side_one),
        }
    }

    pub fn get_both_sides_immutable(&self, side_ref: &SideReference) -> (&Side, &Side) {
        match side_ref {
            SideReference::SideOne => (&self.side_one, &self.side_two),
            SideReference::SideTwo => (&self.side_two, &self.side_one),
        }
    }

    pub fn re_enable_disabled_moves(
        &mut self,
        side_ref: &SideReference,
        vec_to_add_to: &mut Vec<Instruction>,
    ) {
        let active = self.get_side(side_ref).get_active();
        if active.moves.m0.disabled {
            active.moves.m0.disabled = false;
            vec_to_add_to.push(Instruction::EnableMove(EnableMoveInstruction {
                side_ref: *side_ref,
                move_index: PokemonMoveIndex::M0,
            }));
        }
        if active.moves.m1.disabled {
            active.moves.m1.disabled = false;
            vec_to_add_to.push(Instruction::EnableMove(EnableMoveInstruction {
                side_ref: *side_ref,
                move_index: PokemonMoveIndex::M1,
            }));
        }
        if active.moves.m2.disabled {
            active.moves.m2.disabled = false;
            vec_to_add_to.push(Instruction::EnableMove(EnableMoveInstruction {
                side_ref: *side_ref,
                move_index: PokemonMoveIndex::M2,
            }));
        }
        if active.moves.m3.disabled {
            active.moves.m3.disabled = false;
            vec_to_add_to.push(Instruction::EnableMove(EnableMoveInstruction {
                side_ref: *side_ref,
                move_index: PokemonMoveIndex::M3,
            }));
        }
    }

    pub fn reset_toxic_count(
        &mut self,
        side_ref: &SideReference,
        vec_to_add_to: &mut Vec<Instruction>,
    ) {
        let side = self.get_side(side_ref);
        if side.side_conditions.toxic_count > 0 {
            vec_to_add_to.push(Instruction::ChangeSideCondition(
                ChangeSideConditionInstruction {
                    side_ref: *side_ref,
                    side_condition: PokemonSideCondition::ToxicCount,
                    amount: -1 * side.side_conditions.toxic_count,
                },
            ));
            side.side_conditions.toxic_count = 0;
        }
    }

    pub fn remove_volatile_statuses_on_switch(
        &mut self,
        side_ref: &SideReference,
        instructions: &mut Vec<Instruction>,
        baton_passing: bool,
    ) {
        let side = self.get_side(side_ref);

        // Take ownership of the current set to avoid borrow conflicts
        // since we may need to modify the side in the loop
        let mut volatile_statuses = std::mem::take(&mut side.volatile_statuses);

        volatile_statuses.retain(|pkmn_volatile_status| {
            let should_retain = match pkmn_volatile_status {
                PokemonVolatileStatus::SUBSTITUTE => baton_passing,
                PokemonVolatileStatus::LEECHSEED => baton_passing,
                PokemonVolatileStatus::TYPECHANGE => {
                    let active = side.get_active();
                    if active.base_types != active.types {
                        instructions.push(Instruction::ChangeType(ChangeType {
                            side_ref: *side_ref,
                            new_types: active.base_types,
                            old_types: active.types,
                        }));
                        active.types = active.base_types;
                    }
                    false
                }
                // While you can't switch out of a locked move you can be forced out in other ways
                PokemonVolatileStatus::LOCKEDMOVE => {
                    instructions.push(Instruction::ChangeVolatileStatusDuration(
                        ChangeVolatileStatusDurationInstruction {
                            side_ref: *side_ref,
                            volatile_status: *pkmn_volatile_status,
                            amount: -1 * side.volatile_status_durations.lockedmove,
                        },
                    ));
                    side.volatile_status_durations.lockedmove = 0;
                    false
                }
                _ => false,
            };

            if !should_retain {
                instructions.push(Instruction::RemoveVolatileStatus(
                    RemoveVolatileStatusInstruction {
                        side_ref: *side_ref,
                        volatile_status: *pkmn_volatile_status,
                    },
                ));
            }
            should_retain
        });

        // Clean up by re-setting the volatile statuses
        side.volatile_statuses = volatile_statuses;
    }

    pub fn reset_boosts(&mut self, side_ref: &SideReference, vec_to_add_to: &mut Vec<Instruction>) {
        let side = self.get_side(side_ref);

        if side.attack_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::Attack,
                amount: -1 * side.attack_boost,
            }));
            side.attack_boost = 0;
        }

        if side.defense_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::Defense,
                amount: -1 * side.defense_boost,
            }));
            side.defense_boost = 0;
        }

        if side.special_attack_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::SpecialAttack,
                amount: -1 * side.special_attack_boost,
            }));
            side.special_attack_boost = 0;
        }

        if side.special_defense_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::SpecialDefense,
                amount: -1 * side.special_defense_boost,
            }));
            side.special_defense_boost = 0;
        }

        if side.speed_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::Speed,
                amount: -1 * side.speed_boost,
            }));
            side.speed_boost = 0;
        }

        if side.evasion_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::Evasion,
                amount: -1 * side.evasion_boost,
            }));
            side.evasion_boost = 0;
        }

        if side.accuracy_boost != 0 {
            vec_to_add_to.push(Instruction::Boost(BoostInstruction {
                side_ref: *side_ref,
                stat: PokemonBoostableStat::Accuracy,
                amount: -1 * side.accuracy_boost,
            }));
            side.accuracy_boost = 0;
        }
    }

    pub fn terrain_is_active(&self, terrain: &Terrain) -> bool {
        &self.terrain.terrain_type == terrain && self.terrain.turns_remaining > 0
    }

    pub fn weather_is_active(&self, weather: &Weather) -> bool {
        let s1_active = self.side_one.get_active_immutable();
        let s2_active = self.side_two.get_active_immutable();
        &self.weather.weather_type == weather
            && s1_active.ability != Abilities::AIRLOCK
            && s1_active.ability != Abilities::CLOUDNINE
            && s2_active.ability != Abilities::AIRLOCK
            && s2_active.ability != Abilities::CLOUDNINE
    }

    fn _state_contains_any_move(&self, moves: &[Choices]) -> bool {
        for s in [&self.side_one, &self.side_two] {
            for pkmn in s.pokemon.into_iter() {
                for mv in pkmn.moves.into_iter() {
                    if moves.contains(&mv.id) {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub fn set_damage_dealt_flag(&mut self) {
        if self._state_contains_any_move(&[
            Choices::COUNTER,
            Choices::MIRRORCOAT,
            Choices::METALBURST,
            Choices::COMEUPPANCE,
            Choices::FOCUSPUNCH,
        ]) {
            self.use_damage_dealt = true
        }
    }

    pub fn set_last_used_move_flag(&mut self) {
        if self._state_contains_any_move(&[
            Choices::ENCORE,
            Choices::FAKEOUT,
            Choices::FIRSTIMPRESSION,
            Choices::BLOODMOON,
            Choices::GIGATONHAMMER,
        ]) {
            self.use_last_used_move = true
        }
    }

    pub fn set_conditional_mechanics(&mut self) {
        /*
        These mechanics are not always relevant but when they are it
        is important that they are enabled. Enabling them all the time would
        suffer about a 20% performance hit.
        */
        self.set_damage_dealt_flag();
        self.set_last_used_move_flag();
    }

    fn damage(&mut self, side_ref: &SideReference, amount: i16) {
        let active = self.get_side(&side_ref).get_active();

        active.hp -= amount;
    }

    fn heal(&mut self, side_ref: &SideReference, amount: i16) {
        let active = self.get_side(&side_ref).get_active();

        active.hp += amount;
    }

    fn switch(
        &mut self,
        side_ref: &SideReference,
        next_active_index: PokemonIndex,
        _: PokemonIndex,
    ) {
        let side = self.get_side(&side_ref);
        side.active_index = next_active_index;
    }

    fn reverse_switch(
        &mut self,
        side_ref: &SideReference,
        _: PokemonIndex,
        previous_active_index: PokemonIndex,
    ) {
        let side = self.get_side(&side_ref);
        side.active_index = previous_active_index;
    }

    fn apply_volatile_status(
        &mut self,
        side_ref: &SideReference,
        volatile_status: PokemonVolatileStatus,
    ) {
        self.get_side(&side_ref)
            .volatile_statuses
            .insert(volatile_status);
    }

    fn remove_volatile_status(
        &mut self,
        side_ref: &SideReference,
        volatile_status: PokemonVolatileStatus,
    ) {
        self.get_side(&side_ref)
            .volatile_statuses
            .remove(&volatile_status);
    }

    fn change_status(
        &mut self,
        side_ref: &SideReference,
        pokemon_index: PokemonIndex,
        new_status: PokemonStatus,
    ) {
        let pkmn = &mut self.get_side(&side_ref).pokemon[pokemon_index];
        pkmn.status = new_status;
    }

    fn apply_boost(&mut self, side_ref: &SideReference, stat: &PokemonBoostableStat, amount: i8) {
        let side = self.get_side(&side_ref);
        match stat {
            PokemonBoostableStat::Attack => side.attack_boost += amount,
            PokemonBoostableStat::Defense => side.defense_boost += amount,
            PokemonBoostableStat::SpecialAttack => side.special_attack_boost += amount,
            PokemonBoostableStat::SpecialDefense => side.special_defense_boost += amount,
            PokemonBoostableStat::Speed => side.speed_boost += amount,
            PokemonBoostableStat::Evasion => side.evasion_boost += amount,
            PokemonBoostableStat::Accuracy => side.accuracy_boost += amount,
        }
    }

    fn increment_side_condition(
        &mut self,
        side_ref: &SideReference,
        side_condition: &PokemonSideCondition,
        amount: i8,
    ) {
        let side = self.get_side(&side_ref);

        match side_condition {
            PokemonSideCondition::AuroraVeil => side.side_conditions.aurora_veil += amount,
            PokemonSideCondition::CraftyShield => side.side_conditions.crafty_shield += amount,
            PokemonSideCondition::HealingWish => side.side_conditions.healing_wish += amount,
            PokemonSideCondition::LightScreen => side.side_conditions.light_screen += amount,
            PokemonSideCondition::LuckyChant => side.side_conditions.lucky_chant += amount,
            PokemonSideCondition::LunarDance => side.side_conditions.lunar_dance += amount,
            PokemonSideCondition::MatBlock => side.side_conditions.mat_block += amount,
            PokemonSideCondition::Mist => side.side_conditions.mist += amount,
            PokemonSideCondition::Protect => side.side_conditions.protect += amount,
            PokemonSideCondition::QuickGuard => side.side_conditions.quick_guard += amount,
            PokemonSideCondition::Reflect => side.side_conditions.reflect += amount,
            PokemonSideCondition::Safeguard => side.side_conditions.safeguard += amount,
            PokemonSideCondition::Spikes => side.side_conditions.spikes += amount,
            PokemonSideCondition::Stealthrock => side.side_conditions.stealth_rock += amount,
            PokemonSideCondition::StickyWeb => side.side_conditions.sticky_web += amount,
            PokemonSideCondition::Tailwind => side.side_conditions.tailwind += amount,
            PokemonSideCondition::ToxicCount => side.side_conditions.toxic_count += amount,
            PokemonSideCondition::ToxicSpikes => side.side_conditions.toxic_spikes += amount,
            PokemonSideCondition::WideGuard => side.side_conditions.wide_guard += amount,
        }
    }

    fn increment_volatile_status_duration(
        &mut self,
        side_ref: &SideReference,
        volatile_status: &PokemonVolatileStatus,
        amount: i8,
    ) {
        let side = self.get_side(&side_ref);
        match volatile_status {
            PokemonVolatileStatus::CONFUSION => {
                side.volatile_status_durations.confusion += amount;
            }
            PokemonVolatileStatus::LOCKEDMOVE => {
                side.volatile_status_durations.lockedmove += amount;
            }
            PokemonVolatileStatus::ENCORE => {
                side.volatile_status_durations.encore += amount;
            }
            _ => panic!(
                "Invalid volatile status for increment_volatile_status_duration: {:?}",
                volatile_status
            ),
        }
    }

    fn change_types(
        &mut self,
        side_reference: &SideReference,
        new_types: (PokemonType, PokemonType),
    ) {
        self.get_side(side_reference).get_active().types = new_types;
    }

    fn change_item(&mut self, side_reference: &SideReference, new_item: Items) {
        self.get_side(side_reference).get_active().item = new_item;
    }

    fn change_weather(&mut self, weather_type: Weather, turns_remaining: i8) {
        self.weather.weather_type = weather_type;
        self.weather.turns_remaining = turns_remaining;
    }

    fn change_terrain(&mut self, terrain_type: Terrain, turns_remaining: i8) {
        self.terrain.terrain_type = terrain_type;
        self.terrain.turns_remaining = turns_remaining;
    }

    fn enable_move(&mut self, side_reference: &SideReference, move_index: &PokemonMoveIndex) {
        self.get_side(side_reference).get_active().moves[move_index].disabled = false;
    }

    fn disable_move(&mut self, side_reference: &SideReference, move_index: &PokemonMoveIndex) {
        self.get_side(side_reference).get_active().moves[move_index].disabled = true;
    }

    fn set_wish(&mut self, side_reference: &SideReference, wish_amount_change: i16) {
        self.get_side(side_reference).wish.0 = 2;
        self.get_side(side_reference).wish.1 += wish_amount_change;
    }

    fn unset_wish(&mut self, side_reference: &SideReference, wish_amount_change: i16) {
        self.get_side(side_reference).wish.0 = 0;
        self.get_side(side_reference).wish.1 -= wish_amount_change;
    }

    fn increment_wish(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).wish.0 += 1;
    }

    fn decrement_wish(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).wish.0 -= 1;
    }

    fn set_future_sight(&mut self, side_reference: &SideReference, pokemon_index: PokemonIndex) {
        let side = self.get_side(side_reference);
        side.future_sight.0 = 3;
        side.future_sight.1 = pokemon_index;
    }

    fn unset_future_sight(
        &mut self,
        side_reference: &SideReference,
        previous_pokemon_index: PokemonIndex,
    ) {
        let side = self.get_side(side_reference);
        side.future_sight.0 = 0;
        side.future_sight.1 = previous_pokemon_index;
    }

    fn increment_future_sight(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).future_sight.0 += 1;
    }

    fn decrement_future_sight(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).future_sight.0 -= 1;
    }

    fn damage_substitute(&mut self, side_reference: &SideReference, amount: i16) {
        self.get_side(side_reference).substitute_health -= amount;
    }

    fn heal_substitute(&mut self, side_reference: &SideReference, amount: i16) {
        self.get_side(side_reference).substitute_health += amount;
    }

    fn set_substitute_health(&mut self, side_reference: &SideReference, amount: i16) {
        self.get_side(side_reference).substitute_health += amount;
    }

    fn decrement_rest_turn(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).get_active().rest_turns -= 1;
    }

    fn increment_rest_turn(&mut self, side_reference: &SideReference) {
        self.get_side(side_reference).get_active().rest_turns += 1;
    }

    fn set_rest_turn(
        &mut self,
        side_reference: &SideReference,
        pokemon_index: PokemonIndex,
        amount: i8,
    ) {
        self.get_side(side_reference).pokemon[pokemon_index].rest_turns = amount;
    }

    fn set_sleep_turn(
        &mut self,
        side_reference: &SideReference,
        pokemon_index: PokemonIndex,
        amount: i8,
    ) {
        self.get_side(side_reference).pokemon[pokemon_index].sleep_turns = amount;
    }

    fn toggle_trickroom(&mut self, new_turns_remaining: i8) {
        self.trick_room.active = !self.trick_room.active;
        self.trick_room.turns_remaining = new_turns_remaining;
    }

    fn set_last_used_move(&mut self, side_reference: &SideReference, last_used_move: LastUsedMove) {
        match side_reference {
            SideReference::SideOne => self.side_one.last_used_move = last_used_move,
            SideReference::SideTwo => self.side_two.last_used_move = last_used_move,
        }
    }

    fn decrement_pp(
        &mut self,
        side_reference: &SideReference,
        move_index: &PokemonMoveIndex,
        amount: &i8,
    ) {
        match side_reference {
            SideReference::SideOne => self.side_one.get_active().moves[move_index].pp -= amount,
            SideReference::SideTwo => self.side_two.get_active().moves[move_index].pp -= amount,
        }
    }

    fn increment_pp(
        &mut self,
        side_reference: &SideReference,
        move_index: &PokemonMoveIndex,
        amount: &i8,
    ) {
        match side_reference {
            SideReference::SideOne => self.side_one.get_active().moves[move_index].pp += amount,
            SideReference::SideTwo => self.side_two.get_active().moves[move_index].pp += amount,
        }
    }

    fn set_damage_dealt(
        side: &mut Side,
        damage_change: i16,
        move_category: MoveCategory,
        toggle_hit_substitute: bool,
    ) {
        side.damage_dealt.damage += damage_change;
        side.damage_dealt.move_category = move_category;
        if toggle_hit_substitute {
            side.damage_dealt.hit_substitute = !side.damage_dealt.hit_substitute;
        }
    }

    pub fn apply_instructions(&mut self, instructions: &Vec<Instruction>) {
        for i in instructions {
            self.apply_one_instruction(i)
        }
    }

    pub fn apply_one_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Damage(instruction) => {
                self.damage(&instruction.side_ref, instruction.damage_amount)
            }
            Instruction::Switch(instruction) => self.switch(
                &instruction.side_ref,
                instruction.next_index,
                instruction.previous_index,
            ),
            Instruction::ApplyVolatileStatus(instruction) => {
                self.apply_volatile_status(&instruction.side_ref, instruction.volatile_status)
            }
            Instruction::RemoveVolatileStatus(instruction) => {
                self.remove_volatile_status(&instruction.side_ref, instruction.volatile_status)
            }
            Instruction::ChangeStatus(instruction) => self.change_status(
                &instruction.side_ref,
                instruction.pokemon_index,
                instruction.new_status,
            ),
            Instruction::Boost(instruction) => {
                self.apply_boost(&instruction.side_ref, &instruction.stat, instruction.amount)
            }
            Instruction::ChangeSideCondition(instruction) => self.increment_side_condition(
                &instruction.side_ref,
                &instruction.side_condition,
                instruction.amount,
            ),
            Instruction::ChangeVolatileStatusDuration(instruction) => self
                .increment_volatile_status_duration(
                    &instruction.side_ref,
                    &instruction.volatile_status,
                    instruction.amount,
                ),
            Instruction::ChangeWeather(instruction) => self.change_weather(
                instruction.new_weather,
                instruction.new_weather_turns_remaining,
            ),
            Instruction::DecrementWeatherTurnsRemaining => {
                self.weather.turns_remaining -= 1;
            }
            Instruction::ChangeTerrain(instruction) => self.change_terrain(
                instruction.new_terrain,
                instruction.new_terrain_turns_remaining,
            ),
            Instruction::DecrementTerrainTurnsRemaining => {
                self.terrain.turns_remaining -= 1;
            }
            Instruction::ChangeType(instruction) => {
                self.change_types(&instruction.side_ref, instruction.new_types)
            }
            Instruction::ChangeAbility(instruction) => {
                self.get_side(&instruction.side_ref).get_active().ability =
                    instruction.new_ability.clone();
            }
            Instruction::Heal(instruction) => {
                self.heal(&instruction.side_ref, instruction.heal_amount)
            }
            Instruction::ChangeItem(instruction) => {
                self.change_item(&instruction.side_ref, instruction.new_item)
            }
            Instruction::ChangeAttack(instruction) => {
                self.get_side(&instruction.side_ref).get_active().attack += instruction.amount;
            }
            Instruction::ChangeDefense(instruction) => {
                self.get_side(&instruction.side_ref).get_active().defense += instruction.amount;
            }
            Instruction::ChangeSpecialAttack(instruction) => {
                self.get_side(&instruction.side_ref)
                    .get_active()
                    .special_attack += instruction.amount;
            }
            Instruction::ChangeSpecialDefense(instruction) => {
                self.get_side(&instruction.side_ref)
                    .get_active()
                    .special_defense += instruction.amount;
            }
            Instruction::ChangeSpeed(instruction) => {
                self.get_side(&instruction.side_ref).get_active().speed += instruction.amount;
            }
            Instruction::EnableMove(instruction) => {
                self.enable_move(&instruction.side_ref, &instruction.move_index)
            }
            Instruction::DisableMove(instruction) => {
                self.disable_move(&instruction.side_ref, &instruction.move_index)
            }
            Instruction::ChangeWish(instruction) => {
                self.set_wish(&instruction.side_ref, instruction.wish_amount_change);
            }
            Instruction::DecrementWish(instruction) => {
                self.decrement_wish(&instruction.side_ref);
            }
            Instruction::SetFutureSight(instruction) => {
                self.set_future_sight(&instruction.side_ref, instruction.pokemon_index);
            }
            Instruction::DecrementFutureSight(instruction) => {
                self.decrement_future_sight(&instruction.side_ref);
            }
            Instruction::DamageSubstitute(instruction) => {
                self.damage_substitute(&instruction.side_ref, instruction.damage_amount);
            }
            Instruction::ChangeSubstituteHealth(instruction) => {
                self.set_substitute_health(&instruction.side_ref, instruction.health_change);
            }
            Instruction::SetRestTurns(instruction) => {
                self.set_rest_turn(
                    &instruction.side_ref,
                    instruction.pokemon_index,
                    instruction.new_turns,
                );
            }
            Instruction::SetSleepTurns(instruction) => {
                self.set_sleep_turn(
                    &instruction.side_ref,
                    instruction.pokemon_index,
                    instruction.new_turns,
                );
            }
            Instruction::DecrementRestTurns(instruction) => {
                self.decrement_rest_turn(&instruction.side_ref);
            }
            Instruction::ToggleTrickRoom(instruction) => {
                self.toggle_trickroom(instruction.new_trickroom_turns_remaining)
            }
            Instruction::DecrementTrickRoomTurnsRemaining => {
                self.trick_room.turns_remaining -= 1;
            }
            Instruction::ToggleSideOneForceSwitch => self.side_one.toggle_force_switch(),
            Instruction::ToggleSideTwoForceSwitch => self.side_two.toggle_force_switch(),
            Instruction::SetSideOneMoveSecondSwitchOutMove(instruction) => {
                self.side_one.switch_out_move_second_saved_move = instruction.new_choice;
            }
            Instruction::SetSideTwoMoveSecondSwitchOutMove(instruction) => {
                self.side_two.switch_out_move_second_saved_move = instruction.new_choice;
            }
            Instruction::ToggleBatonPassing(instruction) => match instruction.side_ref {
                SideReference::SideOne => {
                    self.side_one.baton_passing = !self.side_one.baton_passing
                }
                SideReference::SideTwo => {
                    self.side_two.baton_passing = !self.side_two.baton_passing
                }
            },
            Instruction::ToggleTerastallized(instruction) => match instruction.side_ref {
                SideReference::SideOne => self.side_one.get_active().terastallized ^= true,
                SideReference::SideTwo => self.side_two.get_active().terastallized ^= true,
            },
            Instruction::SetLastUsedMove(instruction) => {
                self.set_last_used_move(&instruction.side_ref, instruction.last_used_move)
            }
            Instruction::SetDamageDealtSideOne(instruction) => State::set_damage_dealt(
                &mut self.side_one,
                instruction.damage_change,
                instruction.move_category,
                instruction.toggle_hit_substitute,
            ),
            Instruction::SetDamageDealtSideTwo(instruction) => State::set_damage_dealt(
                &mut self.side_two,
                instruction.damage_change,
                instruction.move_category,
                instruction.toggle_hit_substitute,
            ),
            Instruction::DecrementPP(instruction) => self.decrement_pp(
                &instruction.side_ref,
                &instruction.move_index,
                &instruction.amount,
            ),
            Instruction::FormeChange(instruction) => {
                self.get_side(&instruction.side_ref).get_active().id = instruction.new_forme;
            }
        }
    }

    pub fn reverse_instructions(&mut self, instructions: &Vec<Instruction>) {
        for i in instructions.iter().rev() {
            self.reverse_one_instruction(i);
        }
    }

    pub fn reverse_one_instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Damage(instruction) => {
                self.heal(&instruction.side_ref, instruction.damage_amount)
            }
            Instruction::Switch(instruction) => self.reverse_switch(
                &instruction.side_ref,
                instruction.next_index,
                instruction.previous_index,
            ),
            Instruction::ApplyVolatileStatus(instruction) => {
                self.remove_volatile_status(&instruction.side_ref, instruction.volatile_status)
            }
            Instruction::RemoveVolatileStatus(instruction) => {
                self.apply_volatile_status(&instruction.side_ref, instruction.volatile_status)
            }
            Instruction::ChangeStatus(instruction) => self.change_status(
                &instruction.side_ref,
                instruction.pokemon_index,
                instruction.old_status,
            ),
            Instruction::Boost(instruction) => self.apply_boost(
                &instruction.side_ref,
                &instruction.stat,
                -1 * instruction.amount,
            ),
            Instruction::ChangeSideCondition(instruction) => self.increment_side_condition(
                &instruction.side_ref,
                &instruction.side_condition,
                -1 * instruction.amount,
            ),
            Instruction::ChangeVolatileStatusDuration(instruction) => self
                .increment_volatile_status_duration(
                    &instruction.side_ref,
                    &instruction.volatile_status,
                    -1 * instruction.amount,
                ),
            Instruction::ChangeWeather(instruction) => self.change_weather(
                instruction.previous_weather,
                instruction.previous_weather_turns_remaining,
            ),
            Instruction::DecrementWeatherTurnsRemaining => {
                self.weather.turns_remaining += 1;
            }
            Instruction::ChangeTerrain(instruction) => self.change_terrain(
                instruction.previous_terrain,
                instruction.previous_terrain_turns_remaining,
            ),
            Instruction::DecrementTerrainTurnsRemaining => {
                self.terrain.turns_remaining += 1;
            }
            Instruction::ChangeType(instruction) => {
                self.change_types(&instruction.side_ref, instruction.old_types)
            }
            Instruction::ChangeAbility(instruction) => {
                self.get_side(&instruction.side_ref).get_active().ability =
                    instruction.old_ability.clone();
            }
            Instruction::EnableMove(instruction) => {
                self.disable_move(&instruction.side_ref, &instruction.move_index)
            }
            Instruction::DisableMove(instruction) => {
                self.enable_move(&instruction.side_ref, &instruction.move_index)
            }
            Instruction::Heal(instruction) => {
                self.damage(&instruction.side_ref, instruction.heal_amount)
            }
            Instruction::ChangeItem(instruction) => {
                self.change_item(&instruction.side_ref, instruction.current_item)
            }
            Instruction::ChangeAttack(instruction) => {
                self.get_side(&instruction.side_ref).get_active().attack -= instruction.amount;
            }
            Instruction::ChangeDefense(instruction) => {
                self.get_side(&instruction.side_ref).get_active().defense -= instruction.amount;
            }
            Instruction::ChangeSpecialAttack(instruction) => {
                self.get_side(&instruction.side_ref)
                    .get_active()
                    .special_attack -= instruction.amount;
            }
            Instruction::ChangeSpecialDefense(instruction) => {
                self.get_side(&instruction.side_ref)
                    .get_active()
                    .special_defense -= instruction.amount;
            }
            Instruction::ChangeSpeed(instruction) => {
                self.get_side(&instruction.side_ref).get_active().speed -= instruction.amount;
            }
            Instruction::ChangeWish(instruction) => {
                self.unset_wish(&instruction.side_ref, instruction.wish_amount_change)
            }
            Instruction::DecrementWish(instruction) => self.increment_wish(&instruction.side_ref),
            Instruction::SetFutureSight(instruction) => {
                self.unset_future_sight(&instruction.side_ref, instruction.previous_pokemon_index)
            }
            Instruction::DecrementFutureSight(instruction) => {
                self.increment_future_sight(&instruction.side_ref)
            }
            Instruction::DamageSubstitute(instruction) => {
                self.heal_substitute(&instruction.side_ref, instruction.damage_amount);
            }
            Instruction::ChangeSubstituteHealth(instruction) => {
                self.set_substitute_health(&instruction.side_ref, -1 * instruction.health_change);
            }
            Instruction::SetRestTurns(instruction) => {
                self.set_rest_turn(
                    &instruction.side_ref,
                    instruction.pokemon_index,
                    instruction.previous_turns,
                );
            }
            Instruction::SetSleepTurns(instruction) => {
                self.set_sleep_turn(
                    &instruction.side_ref,
                    instruction.pokemon_index,
                    instruction.previous_turns,
                );
            }
            Instruction::DecrementRestTurns(instruction) => {
                self.increment_rest_turn(&instruction.side_ref);
            }
            Instruction::ToggleTrickRoom(instruction) => {
                self.toggle_trickroom(instruction.previous_trickroom_turns_remaining)
            }
            Instruction::DecrementTrickRoomTurnsRemaining => {
                self.trick_room.turns_remaining += 1;
            }
            Instruction::ToggleSideOneForceSwitch => self.side_one.toggle_force_switch(),
            Instruction::ToggleSideTwoForceSwitch => self.side_two.toggle_force_switch(),
            Instruction::SetSideOneMoveSecondSwitchOutMove(instruction) => {
                self.side_one.switch_out_move_second_saved_move = instruction.previous_choice;
            }
            Instruction::SetSideTwoMoveSecondSwitchOutMove(instruction) => {
                self.side_two.switch_out_move_second_saved_move = instruction.previous_choice;
            }
            Instruction::ToggleBatonPassing(instruction) => match instruction.side_ref {
                SideReference::SideOne => {
                    self.side_one.baton_passing = !self.side_one.baton_passing
                }
                SideReference::SideTwo => {
                    self.side_two.baton_passing = !self.side_two.baton_passing
                }
            },
            Instruction::ToggleTerastallized(instruction) => match instruction.side_ref {
                SideReference::SideOne => self.side_one.get_active().terastallized ^= true,
                SideReference::SideTwo => self.side_two.get_active().terastallized ^= true,
            },
            Instruction::SetLastUsedMove(instruction) => {
                self.set_last_used_move(&instruction.side_ref, instruction.previous_last_used_move)
            }
            Instruction::SetDamageDealtSideOne(instruction) => State::set_damage_dealt(
                &mut self.side_one,
                -1 * instruction.damage_change,
                instruction.previous_move_category,
                instruction.toggle_hit_substitute,
            ),
            Instruction::SetDamageDealtSideTwo(instruction) => State::set_damage_dealt(
                &mut self.side_two,
                -1 * instruction.damage_change,
                instruction.previous_move_category,
                instruction.toggle_hit_substitute,
            ),
            Instruction::DecrementPP(instruction) => self.increment_pp(
                &instruction.side_ref,
                &instruction.move_index,
                &instruction.amount,
            ),
            Instruction::FormeChange(instruction) => {
                self.get_side(&instruction.side_ref).get_active().id = instruction.previous_forme;
            }
        }
    }
}

impl Move {
    pub fn serialize(&self) -> String {
        format!("{:?};{};{}", self.id, self.disabled, self.pp)
    }
    pub fn deserialize(serialized: &str) -> Move {
        let split: Vec<&str> = serialized.split(";").collect();
        Move {
            id: Choices::from_str(split[0]).unwrap(),
            disabled: split[1].parse::<bool>().unwrap(),
            pp: split[2].parse::<i8>().unwrap(),
            choice: MOVES
                .get(&Choices::from_str(split[0]).unwrap())
                .unwrap()
                .to_owned(),
        }
    }
}

impl Pokemon {
    pub fn serialize(&self) -> String {
        let evs_str = format!(
            "{};{};{};{};{};{}",
            self.evs.0, self.evs.1, self.evs.2, self.evs.3, self.evs.4, self.evs.5
        );
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.id,
            self.level,
            self.types.0.to_string(),
            self.types.1.to_string(),
            self.base_types.0.to_string(),
            self.base_types.1.to_string(),
            self.hp,
            self.maxhp,
            self.ability.to_string(),
            self.base_ability.to_string(),
            self.item.to_string(),
            self.nature.to_string(),
            evs_str,
            self.attack,
            self.defense,
            self.special_attack,
            self.special_defense,
            self.speed,
            self.status.to_string(),
            self.rest_turns,
            self.sleep_turns,
            self.weight_kg,
            self.moves.m0.serialize(),
            self.moves.m1.serialize(),
            self.moves.m2.serialize(),
            self.moves.m3.serialize(),
            self.terastallized,
            self.tera_type.to_string(),
        )
    }

    pub fn deserialize(serialized: &str) -> Pokemon {
        let split: Vec<&str> = serialized.split(",").collect();
        let evs = if split[12] != "" {
            let mut ev_iter = split[12].split(";");
            (
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
                ev_iter.next().unwrap().parse::<u8>().unwrap(),
            )
        } else {
            (85, 85, 85, 85, 85, 85)
        };
        Pokemon {
            id: PokemonName::from_str(split[0]).unwrap(),
            level: split[1].parse::<i8>().unwrap(),
            types: (
                PokemonType::from_str(split[2]).unwrap(),
                PokemonType::from_str(split[3]).unwrap(),
            ),
            base_types: (
                PokemonType::from_str(split[4]).unwrap(),
                PokemonType::from_str(split[5]).unwrap(),
            ),
            hp: split[6].parse::<i16>().unwrap(),
            maxhp: split[7].parse::<i16>().unwrap(),
            ability: Abilities::from_str(split[8]).unwrap(),
            base_ability: Abilities::from_str(split[9]).unwrap(),
            item: Items::from_str(split[10]).unwrap(),
            nature: PokemonNature::from_str(split[11]).unwrap(),
            evs,
            attack: split[13].parse::<i16>().unwrap(),
            defense: split[14].parse::<i16>().unwrap(),
            special_attack: split[15].parse::<i16>().unwrap(),
            special_defense: split[16].parse::<i16>().unwrap(),
            speed: split[17].parse::<i16>().unwrap(),
            status: PokemonStatus::from_str(split[18]).unwrap(),
            rest_turns: split[19].parse::<i8>().unwrap(),
            sleep_turns: split[20].parse::<i8>().unwrap(),
            weight_kg: split[21].parse::<f32>().unwrap(),
            moves: PokemonMoves {
                m0: Move::deserialize(split[22]),
                m1: Move::deserialize(split[23]),
                m2: Move::deserialize(split[24]),
                m3: Move::deserialize(split[25]),
            },
            terastallized: split[26].parse::<bool>().unwrap(),
            tera_type: PokemonType::from_str(split[27]).unwrap(),
        }
    }
}

impl LastUsedMove {
    pub fn serialize(&self) -> String {
        match self {
            LastUsedMove::Move(move_index) => format!("move:{}", move_index.serialize()),
            LastUsedMove::Switch(pkmn_index) => format!("switch:{}", pkmn_index.serialize()),
            LastUsedMove::None => "move:none".to_string(),
        }
    }
    pub fn deserialize(serialized: &str) -> LastUsedMove {
        let split: Vec<&str> = serialized.split(":").collect();
        match split[0] {
            "move" => {
                if split[1] == "none" {
                    LastUsedMove::None
                } else {
                    LastUsedMove::Move(PokemonMoveIndex::deserialize(split[1]))
                }
            }
            "switch" => LastUsedMove::Switch(PokemonIndex::deserialize(split[1])),
            _ => panic!("Invalid LastUsedMove: {}", serialized),
        }
    }
}

impl PokemonIndex {
    pub fn serialize(&self) -> String {
        match self {
            PokemonIndex::P0 => "0".to_string(),
            PokemonIndex::P1 => "1".to_string(),
            PokemonIndex::P2 => "2".to_string(),
            PokemonIndex::P3 => "3".to_string(),
            PokemonIndex::P4 => "4".to_string(),
            PokemonIndex::P5 => "5".to_string(),
        }
    }

    pub fn deserialize(serialized: &str) -> PokemonIndex {
        match serialized {
            "0" => PokemonIndex::P0,
            "1" => PokemonIndex::P1,
            "2" => PokemonIndex::P2,
            "3" => PokemonIndex::P3,
            "4" => PokemonIndex::P4,
            "5" => PokemonIndex::P5,
            _ => panic!("Invalid PokemonIndex: {}", serialized),
        }
    }
}

impl PokemonMoveIndex {
    pub fn serialize(&self) -> String {
        match self {
            PokemonMoveIndex::M0 => "0".to_string(),
            PokemonMoveIndex::M1 => "1".to_string(),
            PokemonMoveIndex::M2 => "2".to_string(),
            PokemonMoveIndex::M3 => "3".to_string(),
        }
    }

    pub fn deserialize(serialized: &str) -> PokemonMoveIndex {
        match serialized {
            "0" => PokemonMoveIndex::M0,
            "1" => PokemonMoveIndex::M1,
            "2" => PokemonMoveIndex::M2,
            "3" => PokemonMoveIndex::M3,
            _ => panic!("Invalid PokemonMoveIndex: {}", serialized),
        }
    }
}

impl SideConditions {
    pub fn serialize(&self) -> String {
        format!(
            "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}",
            self.aurora_veil,
            self.crafty_shield,
            self.healing_wish,
            self.light_screen,
            self.lucky_chant,
            self.lunar_dance,
            self.mat_block,
            self.mist,
            self.protect,
            self.quick_guard,
            self.reflect,
            self.safeguard,
            self.spikes,
            self.stealth_rock,
            self.sticky_web,
            self.tailwind,
            self.toxic_count,
            self.toxic_spikes,
            self.wide_guard,
        )
    }
    pub fn deserialize(serialized: &str) -> SideConditions {
        let split: Vec<&str> = serialized.split(";").collect();
        SideConditions {
            aurora_veil: split[0].parse::<i8>().unwrap(),
            crafty_shield: split[1].parse::<i8>().unwrap(),
            healing_wish: split[2].parse::<i8>().unwrap(),
            light_screen: split[3].parse::<i8>().unwrap(),
            lucky_chant: split[4].parse::<i8>().unwrap(),
            lunar_dance: split[5].parse::<i8>().unwrap(),
            mat_block: split[6].parse::<i8>().unwrap(),
            mist: split[7].parse::<i8>().unwrap(),
            protect: split[8].parse::<i8>().unwrap(),
            quick_guard: split[9].parse::<i8>().unwrap(),
            reflect: split[10].parse::<i8>().unwrap(),
            safeguard: split[11].parse::<i8>().unwrap(),
            spikes: split[12].parse::<i8>().unwrap(),
            stealth_rock: split[13].parse::<i8>().unwrap(),
            sticky_web: split[14].parse::<i8>().unwrap(),
            tailwind: split[15].parse::<i8>().unwrap(),
            toxic_count: split[16].parse::<i8>().unwrap(),
            toxic_spikes: split[17].parse::<i8>().unwrap(),
            wide_guard: split[18].parse::<i8>().unwrap(),
        }
    }
}

impl Side {
    pub fn serialize(&self) -> String {
        let mut vs_string = String::new();
        for vs in &self.volatile_statuses {
            vs_string.push_str(&vs.to_string());
            vs_string.push_str(":");
        }
        format!(
            "{}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}={}",
            self.pokemon.p0.serialize(),
            self.pokemon.p1.serialize(),
            self.pokemon.p2.serialize(),
            self.pokemon.p3.serialize(),
            self.pokemon.p4.serialize(),
            self.pokemon.p5.serialize(),
            self.active_index.serialize(),
            self.side_conditions.serialize(),
            vs_string,
            self.volatile_status_durations.serialize(),
            self.substitute_health,
            self.attack_boost,
            self.defense_boost,
            self.special_attack_boost,
            self.special_defense_boost,
            self.speed_boost,
            self.accuracy_boost,
            self.evasion_boost,
            self.wish.0,
            self.wish.1,
            self.future_sight.0,
            self.future_sight.1.serialize(),
            self.force_switch,
            self.switch_out_move_second_saved_move.to_string(),
            self.baton_passing,
            self.force_trapped,
            self.last_used_move.serialize(),
            self.slow_uturn_move,
        )
    }
    pub fn deserialize(serialized: &str) -> Side {
        let split: Vec<&str> = serialized.split("=").collect();

        let mut vs_hashset = HashSet::new();
        if split[8] != "" {
            for item in split[8].split(":") {
                vs_hashset.insert(PokemonVolatileStatus::from_str(item).unwrap());
            }
        }
        Side {
            pokemon: SidePokemon {
                p0: Pokemon::deserialize(split[0]),
                p1: Pokemon::deserialize(split[1]),
                p2: Pokemon::deserialize(split[2]),
                p3: Pokemon::deserialize(split[3]),
                p4: Pokemon::deserialize(split[4]),
                p5: Pokemon::deserialize(split[5]),
            },
            active_index: PokemonIndex::deserialize(split[6]),
            side_conditions: SideConditions::deserialize(split[7]),
            volatile_statuses: vs_hashset,
            volatile_status_durations: VolatileStatusDurations::deserialize(split[9]),
            substitute_health: split[10].parse::<i16>().unwrap(),
            attack_boost: split[11].parse::<i8>().unwrap(),
            defense_boost: split[12].parse::<i8>().unwrap(),
            special_attack_boost: split[13].parse::<i8>().unwrap(),
            special_defense_boost: split[14].parse::<i8>().unwrap(),
            speed_boost: split[15].parse::<i8>().unwrap(),
            accuracy_boost: split[16].parse::<i8>().unwrap(),
            evasion_boost: split[17].parse::<i8>().unwrap(),
            wish: (
                split[18].parse::<i8>().unwrap(),
                split[19].parse::<i16>().unwrap(),
            ),
            future_sight: (
                split[20].parse::<i8>().unwrap(),
                PokemonIndex::deserialize(split[21]),
            ),
            force_switch: split[22].parse::<bool>().unwrap(),
            switch_out_move_second_saved_move: Choices::from_str(split[23]).unwrap(),
            baton_passing: split[24].parse::<bool>().unwrap(),
            force_trapped: split[25].parse::<bool>().unwrap(),
            last_used_move: LastUsedMove::deserialize(split[26]),
            damage_dealt: DamageDealt::default(),
            slow_uturn_move: split[27].parse::<bool>().unwrap(),
        }
    }
}

impl StateWeather {
    pub fn serialize(&self) -> String {
        format!("{:?};{}", self.weather_type, self.turns_remaining)
    }
    pub fn deserialize(serialized: &str) -> StateWeather {
        let split: Vec<&str> = serialized.split(";").collect();
        StateWeather {
            weather_type: Weather::from_str(split[0]).unwrap(),
            turns_remaining: split[1].parse::<i8>().unwrap(),
        }
    }
}

impl StateTerrain {
    pub fn serialize(&self) -> String {
        format!("{:?};{}", self.terrain_type, self.turns_remaining)
    }
    pub fn deserialize(serialized: &str) -> StateTerrain {
        let split: Vec<&str> = serialized.split(";").collect();
        StateTerrain {
            terrain_type: Terrain::from_str(split[0]).unwrap(),
            turns_remaining: split[1].parse::<i8>().unwrap(),
        }
    }
}

impl StateTrickRoom {
    pub fn serialize(&self) -> String {
        format!("{};{}", self.active, self.turns_remaining)
    }
    pub fn deserialize(serialized: &str) -> StateTrickRoom {
        let split: Vec<&str> = serialized.split(";").collect();
        StateTrickRoom {
            active: split[0].parse::<bool>().unwrap(),
            turns_remaining: split[1].parse::<i8>().unwrap(),
        }
    }
}

impl State {
    pub fn serialize(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}/{}",
            self.side_one.serialize(),
            self.side_two.serialize(),
            self.weather.serialize(),
            self.terrain.serialize(),
            self.trick_room.serialize(),
            self.team_preview
        )
    }

    /// ```
    ///
    /// /*
    /// This doctest does its best to show the format of the serialized state.
    ///
    /// Roughly, the format for a state is:
    ///     side1/side2/weather/terrain/trick_room/team_preview
    ///
    /// Where the format for a side is:
    ///     p0=p1=p2=p3=p4=p5=active_index=side_conditions=wish0=wish1=force_switch=switch_out_move_second_saved_move=baton_passing=force_trapped=last_used_move=slow_uturn_move
    ///
    /// And the format for a pokemon is:
    ///    id,level,type1,type2,hp,maxhp,ability,item,attack,defense,special_attack,special_defense,speed,attack_boost,defense_boost,special_attack_boost,special_defense_boost,speed_boost,accuracy_boost,evasion_boost,status,substitute_health,rest_turns,weight_kg,volatile_statuses,m0,m1,m2,m3
    ///
    /// There's more to it, follow the code below to see a full example of a serialized state.
    /// */
    ///
    /// if cfg!(feature = "gen2") {
    ///    return;
    /// }
    ///
    /// use poke_engine::abilities::Abilities;
    /// use poke_engine::items::Items;
    /// use poke_engine::pokemon::PokemonName;
    /// use poke_engine::state::State;
    ///
    /// let serialized_state = concat!(
    ///
    /// // SIDE 1
    ///
    /// // POKEMON 1
    ///
    /// // name
    /// "alakazam,",
    ///
    /// // level
    /// "100,",
    ///
    /// // type1
    /// "Psychic,",
    ///
    /// // type2
    /// "Typeless,",
    ///
    /// // base_types 1 and 2. These are needed to revert to the correct type when switching out after being typechanged
    /// "Psychic,",
    /// "Typeless,",
    ///
    /// // hp
    /// "251,",
    ///
    /// // maxhp
    /// "251,",
    ///
    /// // ability
    /// "MAGICGUARD,",
    ///
    /// // base ability. This is needed to revert to the correct ability when switching out after having the ability changed
    /// "MAGICGUARD,",
    ///
    /// // item
    /// "LIFEORB,",
    ///
    /// // nature
    /// "SERIOUS,",
    ///
    /// // EVs split by `;`. Leave blank for default EVs (85 in all)
    /// "252;0;252;0;4;0,",
    /// // ",", left blank for default EVs
    ///
    /// // attack,defense,special attack,special defense,speed
    /// // note these are final stats, not base stats
    /// "121,148,353,206,365,",
    ///
    /// // status
    /// "None,",
    ///
    /// // rest_turns
    /// "0,",
    ///
    /// // sleep_turns
    /// "0,",
    ///
    /// // weight_kg
    /// "25.5,",
    ///
    /// // moves 1 through 4 (move_id;disabled;pp)
    /// "PSYCHIC;false;16,GRASSKNOT;false;32,SHADOWBALL;false;24,HIDDENPOWERFIRE70;false;24,",
    ///
    /// // terastallized
    /// "false,",
    ///
    /// // tera_type
    /// "Normal=",
    ///
    /// // all remaining Pokémon shown in 1 line for brevity
    /// "skarmory,100,Steel,Flying,Steel,Flying,271,271,STURDY,STURDY,CUSTAPBERRY,SERIOUS,,259,316,104,177,262,None,0,0,25.5,STEALTHROCK;false;32,SPIKES;false;32,BRAVEBIRD;false;24,THIEF;false;40,false,Normal=",
    /// "tyranitar,100,Rock,Dark,Rock,Dark,404,404,SANDSTREAM,SANDSTREAM,CHOPLEBERRY,SERIOUS,,305,256,203,327,159,None,0,0,25.5,CRUNCH;false;24,SUPERPOWER;false;8,THUNDERWAVE;false;32,PURSUIT;false;32,false,Normal=",
    /// "mamoswine,100,Ice,Ground,Ice,Ground,362,362,THICKFAT,THICKFAT,NEVERMELTICE,SERIOUS,,392,196,158,176,241,None,0,0,25.5,ICESHARD;false;48,EARTHQUAKE;false;16,SUPERPOWER;false;8,ICICLECRASH;false;16,false,Normal=",
    /// "jellicent,100,Water,Ghost,Water,Ghost,404,404,WATERABSORB,WATERABSORB,AIRBALLOON,SERIOUS,,140,237,206,246,180,None,0,0,25.5,TAUNT;false;32,NIGHTSHADE;false;24,WILLOWISP;false;24,RECOVER;false;16,false,Normal=",
    /// "excadrill,100,Ground,Steel,Ground,Steel,362,362,SANDFORCE,SANDFORCE,CHOICESCARF,SERIOUS,,367,156,122,168,302,None,0,0,25.5,EARTHQUAKE;false;16,IRONHEAD;false;24,ROCKSLIDE;false;16,RAPIDSPIN;false;64,false,Normal=",
    ///
    /// // active-index. This is the index of the active Pokémon in the side's Pokémon array
    /// "0=",
    ///
    /// // side conditions are integers
    /// "0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;=",
    ///
    /// // volatile_statuses (delimited by ":")
    /// "=",
    ///
    /// // some volatile statuses have durations associated with them, delimited by ;
    /// "0;0;0=",
    ///
    /// // substitute_health
    /// "0=",
    ///
    /// // For the active pokemon:
    /// // attack_boost,defense_boost,special attack_boost,special defense_boost,speed_boost,accuracy_boost,evasion_boost
    /// "0=0=0=0=0=0=0=",
    ///
    /// // wish condition is represented by 2 integers, the first is how many wish turns remaining, the second is the amount of HP to heal
    /// "0=",
    /// "0=",
    ///
    /// // future sight is represented by the PokemonIndex of the pokemon that used futuresight, and the number of turns remaining until it hits
    /// "0=",
    /// "0=",
    ///
    /// // a boolean representing if the side is forced to switch
    /// "false=",
    ///
    /// // a 'saved moved' that a pokemon may be waiting to use after the opponent finished their uturn/volt switch/etc.
    /// "NONE=",
    ///
    /// // a b=oolean representing if the side is baton passing
    /// "false=",
    ///
    /// // a boolean representing if the side is force trapped. This is only ever externally provided and never changed by the engine
    /// "false=",
    ///
    /// // last used move is a string that can be either "move:move_name" or "switch:pokemon_index"
    /// "switch:0=",
    ///
    /// // a boolean representing if the side is slow uturning.
    /// // This is only ever set externally. It is used to know if the opposing side has a stored move to use after uturn.
    /// "false/",
    ///
    /// // SIDE 2, all in one line for brevity
    /// "terrakion,100,Rock,Fighting,Rock,Fighting,323,323,JUSTIFIED,JUSTIFIED,FOCUSSASH,SERIOUS,,357,216,163,217,346,None,0,0,25.5,CLOSECOMBAT;false;8,STONEEDGE;false;8,STEALTHROCK;false;32,TAUNT;false;32,false,Normal=lucario,100,Fighting,Steel,Fighting,Steel,281,281,JUSTIFIED,JUSTIFIED,LIFEORB,SERIOUS,,350,176,241,177,279,None,0,0,25.5,CLOSECOMBAT;false;8,EXTREMESPEED;false;8,SWORDSDANCE;false;32,CRUNCH;false;24,false,Normal=breloom,100,Grass,Fighting,Grass,Fighting,262,262,TECHNICIAN,TECHNICIAN,LIFEORB,SERIOUS,,394,196,141,156,239,None,0,0,25.5,MACHPUNCH;false;48,BULLETSEED;false;48,SWORDSDANCE;false;32,LOWSWEEP;false;32,false,Normal=keldeo,100,Water,Fighting,Water,Fighting,323,323,JUSTIFIED,JUSTIFIED,LEFTOVERS,SERIOUS,,163,216,357,217,346,None,0,0,25.5,SECRETSWORD;false;16,HYDROPUMP;false;8,SCALD;false;24,SURF;false;24,false,Normal=conkeldurr,100,Fighting,Typeless,Fighting,Typeless,414,414,GUTS,GUTS,LEFTOVERS,SERIOUS,,416,226,132,167,126,None,0,0,25.5,MACHPUNCH;false;48,DRAINPUNCH;false;16,ICEPUNCH;false;24,THUNDERPUNCH;false;24,false,Normal=toxicroak,100,Poison,Fighting,Poison,Fighting,307,307,DRYSKIN,DRYSKIN,LIFEORB,SERIOUS,,311,166,189,167,295,None,0,0,25.5,DRAINPUNCH;false;16,SUCKERPUNCH;false;8,SWORDSDANCE;false;32,ICEPUNCH;false;24,false,Normal=0=0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;==0;0;0=0=0=0=0=0=0=0=0=0=0=0=0=false=NONE=false=false=switch:0=false/",
    ///
    /// // weather is a string representing the weather type and the number of turns remaining
    /// "none;5/",
    ///
    /// // terrain is a string representing the terrain type and the number of turns remaining
    /// "none;5/",
    ///
    /// // trick room is a boolean representing if trick room is active and the number of turns remaining
    /// "false;5/",
    ///
    /// // team preview is a boolean representing if the team preview is active
    /// "false"
    ///
    /// );
    ///
    /// let state = State::deserialize(serialized_state);
    ///
    /// assert_eq!(state.side_one.get_active_immutable().id, PokemonName::ALAKAZAM);
    /// assert_eq!(state.side_one.get_active_immutable().weight_kg, 25.5);
    /// assert_eq!(state.side_one.substitute_health, 0);
    /// assert_eq!(state.side_two.get_active_immutable().id, PokemonName::TERRAKION);
    /// assert_eq!(state.trick_room.active, false);
    /// assert_eq!(state.team_preview, false);
    ///
    /// #[cfg(not(any(feature = "gen1", feature = "gen2")))]
    /// {
    ///     assert_eq!(state.side_two.get_active_immutable().item, Items::FOCUSSASH);
    ///     assert_eq!(state.side_two.get_active_immutable().ability, Abilities::JUSTIFIED);
    ///     assert_eq!(state.side_one.get_active_immutable().item, Items::LIFEORB);
    ///     assert_eq!(state.side_one.get_active_immutable().ability, Abilities::MAGICGUARD);
    /// }
    ///
    /// // the same state, but all in one line
    /// let serialized_state = "alakazam,100,Psychic,Typeless,Psychic,Typeless,251,251,MAGICGUARD,MAGICGUARD,LIFEORB,SERIOUS,252;0;252;0;4;0,121,148,353,206,365,None,0,0,25.5,PSYCHIC;false;16,GRASSKNOT;false;32,SHADOWBALL;false;24,HIDDENPOWERFIRE70;false;24,false,Normal=skarmory,100,Steel,Flying,Steel,Flying,271,271,STURDY,STURDY,CUSTAPBERRY,SERIOUS,,259,316,104,177,262,None,0,0,25.5,STEALTHROCK;false;32,SPIKES;false;32,BRAVEBIRD;false;24,THIEF;false;40,false,Normal=tyranitar,100,Rock,Dark,Rock,Dark,404,404,SANDSTREAM,SANDSTREAM,CHOPLEBERRY,SERIOUS,,305,256,203,327,159,None,0,0,25.5,CRUNCH;false;24,SUPERPOWER;false;8,THUNDERWAVE;false;32,PURSUIT;false;32,false,Normal=mamoswine,100,Ice,Ground,Ice,Ground,362,362,THICKFAT,THICKFAT,NEVERMELTICE,SERIOUS,,392,196,158,176,241,None,0,0,25.5,ICESHARD;false;48,EARTHQUAKE;false;16,SUPERPOWER;false;8,ICICLECRASH;false;16,false,Normal=jellicent,100,Water,Ghost,Water,Ghost,404,404,WATERABSORB,WATERABSORB,AIRBALLOON,SERIOUS,,140,237,206,246,180,None,0,0,25.5,TAUNT;false;32,NIGHTSHADE;false;24,WILLOWISP;false;24,RECOVER;false;16,false,Normal=excadrill,100,Ground,Steel,Ground,Steel,362,362,SANDFORCE,SANDFORCE,CHOICESCARF,SERIOUS,,367,156,122,168,302,None,0,0,25.5,EARTHQUAKE;false;16,IRONHEAD;false;24,ROCKSLIDE;false;16,RAPIDSPIN;false;64,false,Normal=0=0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;==0;0;0=0=0=0=0=0=0=0=0=0=0=0=0=false=NONE=false=false=switch:0=false/terrakion,100,Rock,Fighting,Rock,Fighting,323,323,JUSTIFIED,JUSTIFIED,FOCUSSASH,SERIOUS,,357,216,163,217,346,None,0,0,25.5,CLOSECOMBAT;false;8,STONEEDGE;false;8,STEALTHROCK;false;32,TAUNT;false;32,false,Normal=lucario,100,Fighting,Steel,Fighting,Steel,281,281,JUSTIFIED,JUSTIFIED,LIFEORB,SERIOUS,,350,176,241,177,279,None,0,0,25.5,CLOSECOMBAT;false;8,EXTREMESPEED;false;8,SWORDSDANCE;false;32,CRUNCH;false;24,false,Normal=breloom,100,Grass,Fighting,Grass,Fighting,262,262,TECHNICIAN,TECHNICIAN,LIFEORB,SERIOUS,,394,196,141,156,239,None,0,0,25.5,MACHPUNCH;false;48,BULLETSEED;false;48,SWORDSDANCE;false;32,LOWSWEEP;false;32,false,Normal=keldeo,100,Water,Fighting,Water,Fighting,323,323,JUSTIFIED,JUSTIFIED,LEFTOVERS,SERIOUS,,163,216,357,217,346,None,0,0,25.5,SECRETSWORD;false;16,HYDROPUMP;false;8,SCALD;false;24,SURF;false;24,false,Normal=conkeldurr,100,Fighting,Typeless,Fighting,Typeless,414,414,GUTS,GUTS,LEFTOVERS,SERIOUS,,416,226,132,167,126,None,0,0,25.5,MACHPUNCH;false;48,DRAINPUNCH;false;16,ICEPUNCH;false;24,THUNDERPUNCH;false;24,false,Normal=toxicroak,100,Poison,Fighting,Poison,Fighting,307,307,DRYSKIN,DRYSKIN,LIFEORB,SERIOUS,,311,166,189,167,295,None,0,0,25.5,DRAINPUNCH;false;16,SUCKERPUNCH;false;8,SWORDSDANCE;false;32,ICEPUNCH;false;24,false,Normal=0=0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;==0;0;0=0=0=0=0=0=0=0=0=0=0=0=0=false=NONE=false=false=switch:0=false/none;5/none;5/false;5/false";
    /// let state2 = State::deserialize(serialized_state);
    /// assert_eq!(state.serialize(), state2.serialize());
    ///
    /// ```
    pub fn deserialize(serialized: &str) -> State {
        let split: Vec<&str> = serialized.split("/").collect();
        let mut state = State {
            side_one: Side::deserialize(split[0]),
            side_two: Side::deserialize(split[1]),
            weather: StateWeather::deserialize(split[2]),
            terrain: StateTerrain::deserialize(split[3]),
            trick_room: StateTrickRoom::deserialize(split[4]),
            team_preview: split[5].parse::<bool>().unwrap(),
            use_damage_dealt: false,
            use_last_used_move: false,
        };
        state.set_conditional_mechanics();
        state
    }
}

impl State {
    pub fn visualize(&self) -> String {
        let mut output = String::new();

        // Battle conditions header
        output.push_str(&self.visualize_battle_conditions());
        output.push_str("\n\n");

        // Side One
        output.push_str("=== Side One ===\n");
        output.push_str(&self.visualize_side(&self.side_one));

        // Side Two
        output.push_str("\n=== Side Two ===\n");
        output.push_str(&self.visualize_side(&self.side_two));

        output
    }

    fn visualize_battle_conditions(&self) -> String {
        let mut conditions = Vec::new();

        // Weather
        if self.weather.weather_type != Weather::NONE {
            conditions.push(format!(
                "Weather: {:?} ({} turns)",
                self.weather.weather_type, self.weather.turns_remaining
            ));
        }

        // Terrain
        if self.terrain.terrain_type != Terrain::NONE {
            conditions.push(format!(
                "Terrain: {:?} ({} turns)",
                self.terrain.terrain_type, self.terrain.turns_remaining
            ));
        }

        // Trick Room
        if self.trick_room.active {
            conditions.push(format!(
                "Trick Room active ({} turns)",
                self.trick_room.turns_remaining
            ));
        }

        if conditions.is_empty() {
            "No active battle conditions".to_string()
        } else {
            conditions.join(" | ")
        }
    }

    fn visualize_side(&self, side: &Side) -> String {
        let mut output = String::new();

        // Active Pokemon
        let active = side.get_active_immutable();
        output.push_str(&format!(
            "Active: {} ({}/{})",
            active.id, active.hp, active.maxhp
        ));

        // Status condition
        if active.status != PokemonStatus::NONE {
            output.push_str(&format!(" [{}]", active.status));
        }

        // Types
        if active.terastallized {
            output.push_str(&format!("\nTypes: Tera-{:?}", active.tera_type));
        } else {
            if active.types.1 == PokemonType::TYPELESS {
                output.push_str(&format!("\nTypes: {:?}", active.types.0));
            } else {
                output.push_str(&format!(
                    "\nTypes: {:?}/{:?}",
                    active.types.0, active.types.1
                ));
            }
        }

        // Ability and Item
        output.push_str(&format!(
            "\nAbility: {} | Item: {}",
            active.ability, active.item
        ));

        // Stats and boosts
        let boosts = self.format_stat_boosts(side);
        output.push_str(&format!("\nBoosts: {}", boosts));

        // Moves
        output.push_str("\nMoves: ");
        let moves: Vec<String> = active
            .moves
            .into_iter()
            .filter(|m| m.id != Choices::NONE)
            .map(|m| {
                format!(
                    "{} (PP: {}{})",
                    m.id,
                    m.pp,
                    if m.disabled { " [disabled]" } else { "" }
                )
            })
            .collect();
        output.push_str(&moves.join(", "));

        // Volatile statuses
        if !side.volatile_statuses.is_empty() {
            output.push_str("\nVolatile Status: ");
            let statuses: Vec<String> = side
                .volatile_statuses
                .iter()
                .map(|s| format!("{:?}", s))
                .collect();
            output.push_str(&statuses.join(", "));
        }

        // Side conditions
        let conditions = self.format_side_conditions(&side.side_conditions);
        if !conditions.is_empty() {
            output.push_str("\nSide Conditions: ");
            output.push_str(&conditions);
        }

        // Reserve Pokemon
        output.push_str("\nReserve: ");
        let reserve: Vec<String> = (0..6)
            .filter_map(|i| {
                let pokemon_index = match i {
                    0 => PokemonIndex::P0,
                    1 => PokemonIndex::P1,
                    2 => PokemonIndex::P2,
                    3 => PokemonIndex::P3,
                    4 => PokemonIndex::P4,
                    5 => PokemonIndex::P5,
                    _ => unreachable!(),
                };
                if pokemon_index != side.active_index {
                    let pokemon = &side.pokemon[pokemon_index];
                    if pokemon.id != PokemonName::NONE {
                        Some(format!("{} ({}/{})", pokemon.id, pokemon.hp, pokemon.maxhp))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        output.push_str(&reserve.join(", "));

        output
    }

    fn format_stat_boosts(&self, side: &Side) -> String {
        let mut boosts = Vec::new();

        if side.attack_boost != 0 {
            boosts.push(format!("Atk: {:+}", side.attack_boost));
        }
        if side.defense_boost != 0 {
            boosts.push(format!("Def: {:+}", side.defense_boost));
        }
        if side.special_attack_boost != 0 {
            boosts.push(format!("SpA: {:+}", side.special_attack_boost));
        }
        if side.special_defense_boost != 0 {
            boosts.push(format!("SpD: {:+}", side.special_defense_boost));
        }
        if side.speed_boost != 0 {
            boosts.push(format!("Spe: {:+}", side.speed_boost));
        }
        if side.accuracy_boost != 0 {
            boosts.push(format!("Acc: {:+}", side.accuracy_boost));
        }
        if side.evasion_boost != 0 {
            boosts.push(format!("Eva: {:+}", side.evasion_boost));
        }

        if boosts.is_empty() {
            "none".to_string()
        } else {
            boosts.join(", ")
        }
    }

    fn format_side_conditions(&self, conditions: &SideConditions) -> String {
        let mut active_conditions = Vec::new();

        if conditions.aurora_veil > 0 {
            active_conditions.push(format!("Aurora Veil ({})", conditions.aurora_veil));
        }
        if conditions.light_screen > 0 {
            active_conditions.push(format!("Light Screen ({})", conditions.light_screen));
        }
        if conditions.reflect > 0 {
            active_conditions.push(format!("Reflect ({})", conditions.reflect));
        }
        if conditions.spikes > 0 {
            active_conditions.push(format!("Spikes ({})", conditions.spikes));
        }
        if conditions.stealth_rock > 0 {
            active_conditions.push("Stealth Rock".to_string());
        }
        if conditions.sticky_web > 0 {
            active_conditions.push("Sticky Web".to_string());
        }
        if conditions.toxic_spikes > 0 {
            active_conditions.push(format!("Toxic Spikes ({})", conditions.toxic_spikes));
        }
        if conditions.tailwind > 0 {
            active_conditions.push(format!("Tailwind ({})", conditions.tailwind));
        }

        active_conditions.join(", ")
    }
}
