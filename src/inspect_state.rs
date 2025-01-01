use crate::choices::Choices;
use crate::observation::{generate_observation, HP_BINS};
use crate::state::{pokemon_index_iter, Pokemon, Side, SideReference, State};
use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;

pub fn visualize_state(state: &State) -> String {
    let mut output = String::new();

    // Weather
    writeln!(&mut output, "=== Weather ===").unwrap();
    writeln!(&mut output, "Type: {:?}", state.weather.weather_type).unwrap();
    writeln!(
        &mut output,
        "Turns remaining: {}",
        state.weather.turns_remaining
    )
    .unwrap();

    // Trick Room
    writeln!(&mut output, "\n=== Trick Room ===").unwrap();
    writeln!(&mut output, "Active: {}", state.trick_room.active).unwrap();
    writeln!(
        &mut output,
        "Turns remaining: {}",
        state.trick_room.turns_remaining
    )
    .unwrap();

    // Side Conditions
    for (side, side_name) in [(&state.side_one, "Side One"), (&state.side_two, "Side Two")].iter() {
        writeln!(&mut output, "\n=== {} Side Conditions ===", side_name).unwrap();
        let sc = &side.side_conditions;
        writeln!(&mut output, "Aurora Veil: {}", sc.aurora_veil).unwrap();
        writeln!(&mut output, "Crafty Shield: {}", sc.crafty_shield).unwrap();
        writeln!(&mut output, "Healing Wish: {}", sc.healing_wish).unwrap();
        writeln!(&mut output, "Light Screen: {}", sc.light_screen).unwrap();
        writeln!(&mut output, "Lucky Chant: {}", sc.lucky_chant).unwrap();
        writeln!(&mut output, "Lunar Dance: {}", sc.lunar_dance).unwrap();
        writeln!(&mut output, "Mat Block: {}", sc.mat_block).unwrap();
        writeln!(&mut output, "Mist: {}", sc.mist).unwrap();
        writeln!(&mut output, "Protect: {}", sc.protect).unwrap();
        writeln!(&mut output, "Quick Guard: {}", sc.quick_guard).unwrap();
        writeln!(&mut output, "Reflect: {}", sc.reflect).unwrap();
        writeln!(&mut output, "Safeguard: {}", sc.safeguard).unwrap();
        writeln!(&mut output, "Spikes: {}", sc.spikes).unwrap();
        writeln!(&mut output, "Stealth Rock: {}", sc.stealth_rock).unwrap();
        writeln!(&mut output, "Sticky Web: {}", sc.sticky_web).unwrap();
        writeln!(&mut output, "Tailwind: {}", sc.tailwind).unwrap();
        writeln!(&mut output, "Toxic Spikes: {}", sc.toxic_spikes).unwrap();
        writeln!(&mut output, "Wide Guard: {}", sc.wide_guard).unwrap();
    }

    // Pokemon Details
    for (side, side_name) in [(&state.side_one, "Side One"), (&state.side_two, "Side Two")].iter() {
        writeln!(&mut output, "\n=== {} Pokemon ===", side_name).unwrap();

        // First show active pokemon
        let active = side.get_active_immutable();
        writeln!(&mut output, "\nActive Pokemon:").unwrap();
        write_pokemon_details(&mut output, active, side);

        // Then show benched pokemon
        writeln!(&mut output, "\nBenched Pokemon:").unwrap();
        for pokemon_index in pokemon_index_iter() {
            if pokemon_index != side.active_index {
                writeln!(&mut output, "\n---").unwrap();
                write_pokemon_details(&mut output, &side.pokemon[pokemon_index], side);
            }
        }
    }

    output
}

fn write_pokemon_details(output: &mut String, pokemon: &Pokemon, side: &Side) {
    writeln!(output, "Species: {:?}", pokemon.id).unwrap();
    writeln!(output, "Ability: {:?}", pokemon.ability).unwrap();
    writeln!(output, "Item: {:?}", pokemon.item).unwrap();
    writeln!(output, "HP: {}/{}", pokemon.hp, pokemon.maxhp).unwrap();
    writeln!(
        output,
        "Types: {:?}, {:?}",
        pokemon.types.0, pokemon.types.1
    )
    .unwrap();
    writeln!(output, "Status: {:?}", pokemon.status).unwrap();

    writeln!(output, "Moves:").unwrap();
    for m in pokemon.moves.into_iter().take(4) {
        if m.id != Choices::NONE {
            writeln!(output, "  - {:?} (PP: {})", m.id, m.pp).unwrap();
        }
    }

    writeln!(output, "Boosts:").unwrap();
    writeln!(output, "  Attack: {}", side.attack_boost).unwrap();
    writeln!(output, "  Defense: {}", side.defense_boost).unwrap();
    writeln!(output, "  Sp. Attack: {}", side.special_attack_boost).unwrap();
    writeln!(output, "  Sp. Defense: {}", side.special_defense_boost).unwrap();
    writeln!(output, "  Speed: {}", side.speed_boost).unwrap();
    writeln!(output, "  Accuracy: {}", side.accuracy_boost).unwrap();

    if !side.volatile_statuses.is_empty() {
        writeln!(output, "Volatile Statuses:").unwrap();
        for status in &side.volatile_statuses {
            writeln!(output, "  - {:?}", status).unwrap();
        }
    }
}

pub fn inspect_observation(obs: &[f32], side_ref: SideReference) -> String {
    let mut output = String::new();
    let mut pos = 0;

    // Weather (14 values total)
    writeln!(&mut output, "=== Weather === ({}:{})", pos, pos + 14).unwrap();

    // Weather type (5 values)
    writeln!(
        &mut output,
        "Weather Type ({}:{}) - [None, Sun, Rain, Sand, Hail]",
        pos,
        pos + 5
    )
    .unwrap();
    write_vector_section(&mut output, &obs[pos..pos + 5]);

    // Turns remaining (9 values)
    writeln!(
        &mut output,
        "Turns Remaining ({}:{}) - [0-8 turns]",
        pos + 5,
        pos + 14
    )
    .unwrap();
    write_vector_section(&mut output, &obs[pos + 5..pos + 14]);

    pos += 14;

    // Trick Room (7 values)
    writeln!(&mut output, "\n=== Trick Room === ({}:{})", pos, pos + 7).unwrap();
    writeln!(&mut output, "Position 0: No trick room").unwrap();
    writeln!(&mut output, "Positions 1-6: Turns remaining").unwrap();
    write_vector_section(&mut output, &obs[pos..pos + 7]);
    pos += 7;

    // Side Conditions
    let side_conditions = [
        ("Aurora Veil", 9, "Turns remaining (0-8)"),
        ("Crafty Shield", 2, "Present (0) or Active (1)"),
        ("Healing Wish", 2, "Present (0) or Active (1)"),
        ("Light Screen", 9, "Turns remaining (0-8)"),
        ("Lucky Chant", 6, "Turns remaining (0-5)"),
        ("Lunar Dance", 2, "Present (0) or Active (1)"),
        ("Mat Block", 2, "Present (0) or Active (1)"),
        ("Mist", 6, "Turns remaining (0-5)"),
        ("Protect", 2, "Present (0) or Active (1)"),
        ("Quick Guard", 2, "Present (0) or Active (1)"),
        ("Reflect", 9, "Turns remaining (0-8)"),
        ("Safeguard", 7, "Turns remaining (0-6)"),
        ("Spikes", 4, "Layers (0-3)"),
        ("Stealth Rock", 2, "Present (0) or Active (1)"),
        ("Sticky Web", 2, "Present (0) or Active (1)"),
        ("Tailwind", 5, "Turns remaining (0-4)"),
        ("Toxic Spikes", 3, "Layers (0-2)"),
        ("Wide Guard", 2, "Present (0) or Active (1)"),
    ];

    writeln!(&mut output, "\n=== Our Side Conditions ===").unwrap();
    for (name, size, description) in side_conditions.iter() {
        writeln!(
            &mut output,
            "\n{} ({}:{}) - {}",
            name,
            pos,
            pos + size,
            description
        )
        .unwrap();
        write_vector_section(&mut output, &obs[pos..pos + size]);
        pos += size;
    }

    // Our Side Volatile Statuses (104 values)
    writeln!(
        &mut output,
        "\n=== Our Side Volatile Statuses === ({}:{})",
        pos,
        pos + 104
    )
    .unwrap();
    write_volatile_status_section(&mut output, &obs[pos..pos + 104]);
    pos += 104;

    writeln!(&mut output, "\n=== Opponent Side Conditions ===").unwrap();
    for (name, size, description) in side_conditions.iter() {
        writeln!(
            &mut output,
            "\n{} ({}:{}) - {}",
            name,
            pos,
            pos + size,
            description
        )
        .unwrap();
        write_vector_section(&mut output, &obs[pos..pos + size]);
        pos += size;
    }

    // Opponent Side Volatile Statuses (104 values)
    writeln!(
        &mut output,
        "\n=== Opponent Side Volatile Statuses === ({}:{})",
        pos,
        pos + 104
    )
    .unwrap();
    write_volatile_status_section(&mut output, &obs[pos..pos + 104]);
    pos += 104;

    // Pokemon encoding inspection
    for i in 0..12 {
        let is_active = i < 2;
        let is_our_side = match side_ref {
            SideReference::SideOne => i % 2 == 0,
            SideReference::SideTwo => i % 2 == 1,
        };

        writeln!(
            &mut output,
            "\n=== Pokemon {} ({} {}) ===",
            i,
            if is_our_side { "Our" } else { "Opponent" },
            if is_active { "Active" } else { "Benched" }
        )
        .unwrap();

        let sections = [
            ("Active Flag", 2, "Active (1) or Benched (0)"),
            ("Species", 1, "Pokemon ID number"),
            ("Ability", 1, "Ability ID number"),
            ("Item", 1, "Item ID number"),
            ("Move 1 ID", 1, "Move ID number"),
            ("Move 1 PP", 4, "PP bins (0-3)"),
            ("Move 2 ID", 1, "Move ID number"),
            ("Move 2 PP", 4, "PP bins (0-3)"),
            ("Move 3 ID", 1, "Move ID number"),
            ("Move 3 PP", 4, "PP bins (0-3)"),
            ("Move 4 ID", 1, "Move ID number"),
            ("Move 4 PP", 4, "PP bins (0-3)"),
            ("Types", 20, "One-hot encoded types (positions 0-19)"),
            (
                "HP",
                HP_BINS + 1,
                "Position 0: Fainted, 1-16: HP fraction bins",
            ),
            ("Attack Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Defense Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Sp. Attack Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Sp. Defense Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Speed Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Accuracy Boost", 13, "Positions 0-12: Boost -6 to +6"),
            ("Status", 7, "Position 0: None, 1: Burn, 2: Sleep, etc."),
            ("Rest Turns", 4, "Turns of rest remaining (0-3)"),
            ("Sleep Turns", 4, "Turns of sleep remaining (0-3)"),
        ];

        for (name, size, description) in sections.iter() {
            writeln!(
                &mut output,
                "\n{} ({}:{}) - {}",
                name,
                pos,
                pos + size,
                description
            )
            .unwrap();
            write_vector_section(&mut output, &obs[pos..pos + size]);
            pos += size;
        }
    }

    writeln!(&mut output, "\nTotal length: {}", pos).unwrap();
    output
}

fn write_vector_section(output: &mut String, section: &[f32]) {
    writeln!(output, "Values: {:?}", section).unwrap();
    if let Some(one_hot_pos) = section.iter().position(|&x| x == 1.0) {
        writeln!(output, "One-hot position: {}", one_hot_pos).unwrap();
    }
}

fn write_volatile_status_section(output: &mut String, section: &[f32]) {
    writeln!(output, "Values: {:?}", section).unwrap();
    writeln!(output, "Active volatile statuses (by index):").unwrap();
    for (i, &value) in section.iter().enumerate() {
        if value == 1.0 {
            // For now, just print the index since we don't have TryFrom
            writeln!(output, "  - Status at index {}", i).unwrap();
        }
    }
}

pub fn inspect_state_and_observation(state_str: &str, output_file: &str) -> std::io::Result<()> {
    let state = State::deserialize(state_str);
    let mut file = File::create(output_file)?;

    writeln!(file, "=== STATE VISUALIZATION ===\n")?;
    writeln!(file, "{}", visualize_state(&state))?;

    writeln!(file, "\n=== OBSERVATION ENCODING ===\n")?;
    let obs = generate_observation(&state, SideReference::SideOne);
    writeln!(
        file,
        "{}",
        inspect_observation(&obs, SideReference::SideOne)
    )?;

    Ok(())
}
