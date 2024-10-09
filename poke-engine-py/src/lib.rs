use pyo3::prelude::*;
use pyo3::{pyfunction, pymethods, pymodule, wrap_pyfunction, Bound, PyResult};
use std::collections::HashSet;

use poke_engine::abilities::Abilities;
use poke_engine::choices::{Choices, MOVES};
use poke_engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move_pair,
};
use poke_engine::instruction::StateInstructions;
use poke_engine::io::io_get_all_options;
use poke_engine::items::Items;
use poke_engine::mcts::{perform_mcts, MctsResult, MctsSideResult};
use poke_engine::search::iterative_deepen_expectiminimax;
use poke_engine::state::{
    DamageDealt, LastUsedMove, Move, MoveChoice, Pokemon, PokemonIndex, PokemonMoves,
    PokemonStatus, PokemonType, PokemonVolatileStatus, Side, SideConditions, SidePokemon, State,
    StateTerrain, StateTrickRoom, StateWeather, Terrain, Weather,
};
use std::str::FromStr;
use std::time::Duration;

pub mod pettingzoo;

#[derive(Clone)]
#[pyclass(name = "State")]
pub struct PyState {
    pub state: State,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyState {
    #[new]
    fn new(
        side_one: PySide,
        side_two: PySide,
        weather: String,
        weather_turns_remaining: i8,
        terrain: String,
        terrain_turns_remaining: i8,
        trick_room: bool,
        trick_room_turns_remaining: i8,
        team_preview: bool,
    ) -> Self {
        Self {
            state: State {
                side_one: side_one.create_side(),
                side_two: side_two.create_side(),
                weather: StateWeather {
                    weather_type: Weather::from_str(&weather).unwrap(),
                    turns_remaining: weather_turns_remaining,
                },
                terrain: StateTerrain {
                    terrain_type: Terrain::from_str(&terrain).unwrap(),
                    turns_remaining: terrain_turns_remaining,
                },
                trick_room: StateTrickRoom {
                    active: trick_room,
                    turns_remaining: trick_room_turns_remaining,
                },
                team_preview,
            },
        }
    }

    #[allow(clippy::inherent_to_string)]
    fn to_string(&self) -> String {
        self.state.serialize()
    }

    fn battle_is_over(&self) -> f32 {
        self.state.battle_is_over()
    }

    fn get_all_options(&self) -> (Vec<PyMoveChoice>, Vec<PyMoveChoice>) {
        fn convert_move_choice(choice: MoveChoice, side: &Side) -> PyMoveChoice {
            match choice {
                MoveChoice::Move(m) => PyMoveChoice::Move(
                    side.get_active_immutable().moves[m]
                        .id
                        .to_string()
                        .to_lowercase(),
                ),
                MoveChoice::Switch(s) => PyMoveChoice::Switch(s.into()),
                MoveChoice::None => PyMoveChoice::None(),
            }
        }

        let (side1, side2) = self.state.get_all_options();

        (
            side1
                .into_iter()
                .map(|c| convert_move_choice(c, &self.state.side_one))
                .collect::<Vec<PyMoveChoice>>(),
            side2
                .into_iter()
                .map(|c| convert_move_choice(c, &self.state.side_two))
                .collect::<Vec<PyMoveChoice>>(),
        )
    }
}

#[derive(Clone)]
#[pyclass(name = "Side")]
pub struct PySide {
    pub side: Side,
}

impl PySide {
    fn create_side(&self) -> Side {
        self.side.clone()
    }
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
#[pymethods]
impl PySide {
    #[new]
    fn new(
        active_index: String,
        baton_passing: bool,
        mut pokemon: Vec<PyPokemon>,
        side_conditions: PySideConditions,
        wish: (i8, i16),
        force_switch: bool,
        force_trapped: bool,
        slow_uturn_move: bool,
        volatile_statuses: Vec<String>,
        substitute_health: i16,
        attack_boost: i8,
        defense_boost: i8,
        special_attack_boost: i8,
        special_defense_boost: i8,
        speed_boost: i8,
        accuracy_boost: i8,
        evasion_boost: i8,
        last_used_move: String,
        switch_out_move_second_saved_move: String,
    ) -> Self {
        let mut vs_hashset = HashSet::new();
        for vs in volatile_statuses {
            vs_hashset.insert(PokemonVolatileStatus::deserialize(&vs));
        }

        let remaining_pkmn = 6 - pokemon.len();
        for _ in 0..remaining_pkmn {
            pokemon.push(PyPokemon::create_fainted());
        }

        Self {
            side: Side {
                active_index: PokemonIndex::deserialize(&active_index),
                baton_passing,
                pokemon: SidePokemon {
                    p0: pokemon[0].create_pokemon(),
                    p1: pokemon[1].create_pokemon(),
                    p2: pokemon[2].create_pokemon(),
                    p3: pokemon[3].create_pokemon(),
                    p4: pokemon[4].create_pokemon(),
                    p5: pokemon[5].create_pokemon(),
                },
                side_conditions: side_conditions.create_side_conditions(),
                wish,
                force_switch,
                force_trapped,
                slow_uturn_move,
                volatile_statuses: vs_hashset,
                substitute_health,
                attack_boost,
                defense_boost,
                special_attack_boost,
                special_defense_boost,
                speed_boost,
                accuracy_boost,
                evasion_boost,
                last_used_move: LastUsedMove::deserialize(&last_used_move),
                damage_dealt: DamageDealt::default(),
                switch_out_move_second_saved_move: Choices::from_str(
                    &switch_out_move_second_saved_move,
                )
                .unwrap(),
            },
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "SideConditions")]
pub struct PySideConditions {
    pub side_conditions: SideConditions,
}

impl PySideConditions {
    fn create_side_conditions(&self) -> SideConditions {
        self.side_conditions.clone()
    }
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
#[pymethods]
impl PySideConditions {
    #[new]
    const fn new(
        spikes: i8,
        toxic_spikes: i8,
        stealth_rock: i8,
        sticky_web: i8,
        tailwind: i8,
        lucky_chant: i8,
        lunar_dance: i8,
        reflect: i8,
        light_screen: i8,
        aurora_veil: i8,
        crafty_shield: i8,
        safeguard: i8,
        mist: i8,
        protect: i8,
        healing_wish: i8,
        mat_block: i8,
        quick_guard: i8,
        toxic_count: i8,
        wide_guard: i8,
    ) -> Self {
        Self {
            side_conditions: SideConditions {
                spikes,
                toxic_spikes,
                stealth_rock,
                sticky_web,
                tailwind,
                lucky_chant,
                lunar_dance,
                reflect,
                light_screen,
                aurora_veil,
                crafty_shield,
                safeguard,
                mist,
                protect,
                healing_wish,
                mat_block,
                quick_guard,
                toxic_count,
                wide_guard,
            },
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "Pokemon")]
pub struct PyPokemon {
    pub pokemon: Pokemon,
}

impl PyPokemon {
    fn create_pokemon(&self) -> Pokemon {
        self.pokemon.clone()
    }
    fn create_fainted() -> Self {
        Self {
            pokemon: Pokemon {
                hp: 0,
                ..Default::default()
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyPokemon {
    #[new]
    fn new(
        id: String,
        pokedex_num: i16,
        level: i8,
        types: [String; 2],
        hp: i16,
        maxhp: i16,
        ability: String,
        item: String,
        attack: i16,
        defense: i16,
        special_attack: i16,
        special_defense: i16,
        speed: i16,
        status: String,
        rest_turns: i8,
        sleep_turns: i8,
        weight_kg: f32,
        mut moves: Vec<PyMove>,
    ) -> Self {
        let remaining_pkmn = 6 - moves.len();
        for _ in 0..remaining_pkmn {
            moves.push(PyMove::create_empty_move());
        }
        Self {
            pokemon: Pokemon {
                id,
                pokedex_num,
                level,
                types: (
                    PokemonType::deserialize(&types[0]),
                    PokemonType::deserialize(&types[1]),
                ),
                hp,
                maxhp,
                ability: Abilities::from_str(&ability).unwrap(),
                item: Items::from_str(&item).unwrap(),
                attack,
                defense,
                special_attack,
                special_defense,
                speed,
                status: PokemonStatus::deserialize(&status),
                rest_turns,
                sleep_turns,
                weight_kg,
                moves: PokemonMoves {
                    m0: moves[0].create_move(),
                    m1: moves[1].create_move(),
                    m2: moves[2].create_move(),
                    m3: moves[3].create_move(),
                    m4: moves[4].create_move(),
                    m5: moves[5].create_move(),
                },
            },
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "Move")]
pub struct PyMove {
    pub mv: Move,
}

impl PyMove {
    fn create_move(&self) -> Move {
        self.mv.clone()
    }
    fn create_empty_move() -> Self {
        Self {
            mv: Move {
                disabled: true,
                pp: 0,
                ..Default::default()
            },
        }
    }
}

#[pymethods]
impl PyMove {
    #[new]
    fn new(id: String, move_num: i16, pp: i8, disabled: bool) -> Self {
        let choice = Choices::from_str(&id).unwrap();
        Self {
            mv: Move {
                id: choice,
                move_num,
                disabled,
                pp,
                choice: MOVES.get(&choice).unwrap().clone(),
            },
        }
    }
}

#[derive(Debug)]
#[pyclass(name = "MoveChoice")]
pub enum PyMoveChoice {
    Move(String),
    Switch(u8),
    None(),
}

#[pymethods]
impl PyMoveChoice {
    fn __repr__(&self) -> String {
        match self {
            Self::Move(m) => format!("Move {m}"),
            Self::Switch(s) => format!("Switch {s}"),
            Self::None() => "None".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            Self::Move(m) => m.to_string(),
            Self::Switch(s) => format!("switch {s}"),
            Self::None() => "None".to_string(),
        }
    }
}

#[derive(Clone)]
#[pyclass(get_all)]
struct PyMctsSideResult {
    pub move_choice: String,
    pub total_score: f32,
    pub visits: i64,
}

impl PyMctsSideResult {
    fn from_mcts_side_result(result: MctsSideResult, side: &Side) -> Self {
        Self {
            move_choice: result.move_choice.to_string(side),
            total_score: result.total_score,
            visits: result.visits,
        }
    }
}

#[derive(Clone)]
#[pyclass(get_all)]
struct PyMctsResult {
    s1: Vec<PyMctsSideResult>,
    s2: Vec<PyMctsSideResult>,
    iteration_count: i64,
}

impl PyMctsResult {
    fn from_mcts_result(result: MctsResult, state: &State) -> Self {
        Self {
            s1: result
                .s1
                .iter()
                .map(|r| PyMctsSideResult::from_mcts_side_result(r.clone(), &state.side_one))
                .collect(),
            s2: result
                .s2
                .iter()
                .map(|r| PyMctsSideResult::from_mcts_side_result(r.clone(), &state.side_two))
                .collect(),
            iteration_count: result.iteration_count,
        }
    }
}

#[derive(Clone)]
#[pyclass(get_all)]
struct PyIterativeDeepeningResult {
    s1: Vec<String>,
    s2: Vec<String>,
    matrix: Vec<f32>,
    depth_searched: i8,
}

impl PyIterativeDeepeningResult {
    fn from_iterative_deepening_result(
        result: (Vec<MoveChoice>, Vec<MoveChoice>, Vec<f32>, i8),
        state: &State,
    ) -> Self {
        PyIterativeDeepeningResult {
            s1: result
                .0
                .iter()
                .map(|c| c.to_string(&state.side_one))
                .collect(),
            s2: result
                .1
                .iter()
                .map(|c| c.to_string(&state.side_two))
                .collect(),
            matrix: result.2,
            depth_searched: result.3,
        }
    }
}

#[pyfunction]
fn mcts(mut py_state: PyState, duration_ms: u64) -> PyResult<PyMctsResult> {
    let duration = Duration::from_millis(duration_ms);
    let (s1_options, s2_options) = io_get_all_options(&py_state.state);
    let mcts_result = perform_mcts(&mut py_state.state, s1_options, s2_options, duration);

    let py_mcts_result = PyMctsResult::from_mcts_result(mcts_result, &py_state.state);
    Ok(py_mcts_result)
}

#[pyfunction]
fn id(mut py_state: PyState, duration_ms: u64) -> PyResult<PyIterativeDeepeningResult> {
    let duration = Duration::from_millis(duration_ms);
    let (s1_options, s2_options) = io_get_all_options(&py_state.state);
    let id_result =
        iterative_deepen_expectiminimax(&mut py_state.state, s1_options, s2_options, duration);

    let py_id_result =
        PyIterativeDeepeningResult::from_iterative_deepening_result(id_result, &py_state.state);
    Ok(py_id_result)
}

#[derive(Clone)]
#[pyclass(get_all, set_all)]
struct PyStateInstructions {
    pub percentage: f32,
    pub instruction_list: Vec<String>,
}

impl PyStateInstructions {
    fn from_state_instructions(instructions: &StateInstructions) -> Self {
        Self {
            percentage: instructions.percentage,
            instruction_list: instructions
                .instruction_list
                .iter()
                .map(|i| format!("{i:?}"))
                .collect(),
        }
    }
}

#[pyfunction]
fn gi(
    mut py_state: PyState,
    side_one_move: String,
    side_two_move: String,
) -> PyResult<Vec<PyStateInstructions>> {
    let (s1_move, s2_move);
    match py_state.state.side_one.string_to_movechoice(&side_one_move) {
        Some(m) => s1_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {side_one_move}"
            )))
        }
    }
    match py_state.state.side_two.string_to_movechoice(&side_two_move) {
        Some(m) => s2_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {side_two_move}"
            )))
        }
    }
    let instructions =
        generate_instructions_from_move_pair(&mut py_state.state, &s1_move, &s2_move);
    let py_instructions = instructions
        .iter()
        .map(PyStateInstructions::from_state_instructions)
        .collect();

    Ok(py_instructions)
}

#[pyfunction]
fn calculate_damage(
    py_state: PyState,
    side_one_move: String,
    side_two_move: String,
    side_one_moves_first: bool,
) -> PyResult<(Vec<i16>, Vec<i16>)> {
    let (s1_choice, s2_choice);
    match MOVES.get(&Choices::from_str(side_one_move.as_str()).unwrap()) {
        Some(m) => s1_choice = m.to_owned(),
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {side_one_move}"
            )))
        }
    }
    match MOVES.get(&Choices::from_str(side_two_move.as_str()).unwrap()) {
        Some(m) => s2_choice = m.to_owned(),
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {side_one_move}"
            )))
        }
    }

    let (s1_damage_rolls, s2_damage_rolls) =
        calculate_both_damage_rolls(&py_state.state, s1_choice, s2_choice, side_one_moves_first);

    let (s1_py_rolls, s2_py_rolls);
    match s1_damage_rolls {
        Some(rolls) => s1_py_rolls = rolls,
        None => s1_py_rolls = vec![0],
    }
    match s2_damage_rolls {
        Some(rolls) => s2_py_rolls = rolls,
        None => s2_py_rolls = vec![0],
    }

    Ok((s1_py_rolls, s2_py_rolls))
}

#[pyfunction]
fn state_from_string(s: String) -> PyResult<PyState> {
    Ok(PyState {
        state: State::deserialize(&s),
    })
}

#[pymodule]
#[pyo3(name = "_poke_engine")]
fn py_poke_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(state_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_damage, m)?)?;
    m.add_function(wrap_pyfunction!(gi, m)?)?;
    m.add_function(wrap_pyfunction!(id, m)?)?;
    m.add_function(wrap_pyfunction!(mcts, m)?)?;
    m.add_class::<PyState>()?;
    m.add_class::<PySide>()?;
    m.add_class::<PySideConditions>()?;
    m.add_class::<PyPokemon>()?;
    m.add_class::<PyMove>()?;

    m.add_function(wrap_pyfunction!(pettingzoo::observations, m)?)?;
    Ok(())
}
