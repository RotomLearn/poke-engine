use poke_engine::state::PokemonStatus;
use pyo3::prelude::*;
use pyo3::pyfunction;

use crate::PyState;

#[must_use]
#[pyfunction]
pub fn observations(py_state: &PyState) -> Vec<Vec<PyObject>> {
    let mut res = vec![vec![], vec![]];

    Python::with_gil(|py| {
        // generates a list of observations for each side
        // this will change after considering what each side actually can know about the other side
        for (i, side) in [
            &py_state.state.side_one,
            &py_state.state.side_two,
            &py_state.state.side_two,
            &py_state.state.side_one,
        ]
        .iter()
        .enumerate()
        {
            let side_num = i % 2;

            res[side_num].push((f32::from(side.attack_boost) / 6.0).to_object(py));
            res[side_num].push((f32::from(side.defense_boost) / 6.0).to_object(py));
            res[side_num].push((f32::from(side.evasion_boost) / 6.0).to_object(py));
            res[side_num].push((f32::from(side.special_attack_boost) / 6.0).to_object(py));
            res[side_num].push((f32::from(side.special_defense_boost) / 6.0).to_object(py));
            res[side_num].push((f32::from(side.speed_boost) / 6.0).to_object(py));

            for p in &side.pokemon {
                // 0 for pokemon on this side, 1 for pokemon on other side
                res[side_num].push((i / 2).to_object(py));

                res[side_num].push(p.pokedex_num.to_object(py));
                res[side_num].push((f32::from(p.hp) / f32::from(p.maxhp)).to_object(py));

                res[side_num].push(i32::from(p.status == PokemonStatus::Burn).to_object(py));
                res[side_num].push(i32::from(p.status == PokemonStatus::Sleep).to_object(py));
                res[side_num].push(i32::from(p.status == PokemonStatus::Freeze).to_object(py));
                res[side_num].push(i32::from(p.status == PokemonStatus::Paralyze).to_object(py));
                res[side_num].push(i32::from(p.status == PokemonStatus::Poison).to_object(py));
                res[side_num].push(i32::from(p.status == PokemonStatus::Toxic).to_object(py));
            }
        }
    });

    res
}
