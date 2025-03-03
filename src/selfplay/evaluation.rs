// This file should go in poke_engine/src/selfplay/evaluation.rs

use crate::abilities::ability_on_switch_in;
use crate::generate_instructions::generate_instructions_from_move_pair;
use crate::instruction::{Instruction, StateInstructions};
use crate::items::item_on_switch_in;
use crate::selfplay::player::Player;
use crate::state::{MoveChoice, SideReference, State};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Results from an evaluation match
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub winner: i8,         // 1 for player1, -1 for player2, 0 for draw
    pub turns: u32,         // Number of turns the battle lasted
    pub duration_ms: u64,   // Time taken for the battle in milliseconds
    pub player1_moves: u32, // Number of moves made by player1
    pub player2_moves: u32, // Number of moves made by player2
}

/// Battle implementation specialized for evaluation without training data collection
pub struct EvaluationBattle {
    state: State,
    player1: Box<dyn Player + Send>,
    player2: Box<dyn Player + Send>,
    turn_count: u32,
    player1_move_count: u32,
    player2_move_count: u32,
    start_time: Instant,
    log_file: Option<File>,
}

impl EvaluationBattle {
    pub fn new(
        initial_state: State,
        player1: Box<dyn Player + Send>,
        player2: Box<dyn Player + Send>,
        log_path: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let log_file = if let Some(path) = log_path {
            Some(
                OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?,
            )
        } else {
            None
        };

        Ok(Self {
            state: initial_state,
            player1,
            player2,
            turn_count: 0,
            player1_move_count: 0,
            player2_move_count: 0,
            start_time: Instant::now(),
            log_file,
        })
    }

    fn log(&mut self, content: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "{}", content)?;
            file.flush()?;
        }
        Ok(())
    }

    fn log_state(&mut self, prefix: &str) -> io::Result<()> {
        if let Some(file) = &mut self.log_file {
            writeln!(file, "\n{}", prefix)?;
            writeln!(file, "{}", self.state.serialize())?;
            writeln!(file, "\n")?;
            writeln!(file, "{}", self.state.visualize())?;
            file.flush()?;
        }
        Ok(())
    }

    fn format_move(&self, move_choice: &MoveChoice, is_player1: bool) -> String {
        let side = if is_player1 {
            &self.state.side_one
        } else {
            &self.state.side_two
        };

        match move_choice {
            MoveChoice::Move(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("{}", active_pokemon.moves[m].id)
            }
            MoveChoice::MoveTera(m) => {
                let active_pokemon = side.get_active_immutable();
                format!("Terastallize + {}", active_pokemon.moves[m].id)
            }
            MoveChoice::Switch(p) => {
                format!("Switch to Pokemon at index {:?}", p)
            }
            MoveChoice::None => "None".to_string(),
        }
    }

    fn sample_instructions(
        &mut self,
        move_instructions: Vec<StateInstructions>,
    ) -> Option<StateInstructions> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        let weights: Vec<f64> = move_instructions
            .iter()
            .map(|inst| inst.percentage as f64)
            .collect();

        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        let chosen_idx = dist.sample(&mut rng);

        Some(move_instructions[chosen_idx].clone())
    }

    fn generate_initial_switch_in_instructions(&mut self, state: &mut State) -> Vec<Instruction> {
        let mut final_instructions = Vec::new();
        let mut incoming_instructions = StateInstructions::default();

        // Weather/Terrain setters and other simultaneous effects first
        ability_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        ability_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Then items
        item_on_switch_in(state, &SideReference::SideOne, &mut incoming_instructions);
        item_on_switch_in(state, &SideReference::SideTwo, &mut incoming_instructions);
        final_instructions.extend(incoming_instructions.instruction_list.drain(..));

        // Reverse instructions since we'll apply them later
        state.reverse_instructions(&final_instructions);

        final_instructions
    }

    pub fn play_game(&mut self) -> io::Result<EvaluationResult> {
        self.start_time = Instant::now();

        // Apply initial switch-in instructions
        let mut state_clone = self.state.clone();
        let initial_instructions = self.generate_initial_switch_in_instructions(&mut state_clone);

        if let Some(file) = &mut self.log_file {
            for (j, instruction) in initial_instructions.iter().enumerate() {
                writeln!(file, "  {}: {:?}", j + 1, instruction)?;
            }
        }

        self.state.apply_instructions(&initial_instructions);
        self.log_state("Initial State:")?;

        // Reserialize to ensure clean state
        self.state = State::deserialize(self.state.serialize().as_str());

        while self.state.battle_is_over() == 0.0 {
            self.turn_count += 1;
            self.log(&format!("\n=== Turn {} ===", self.turn_count))?;

            let (side_one_options, side_two_options) = self.state.get_all_options();

            // Format move strings for logging
            let p1_move_strings: Vec<_> = side_one_options
                .iter()
                .map(|m| self.format_move(m, true))
                .collect();
            let p2_move_strings: Vec<_> = side_two_options
                .iter()
                .map(|m| self.format_move(m, false))
                .collect();

            // Log available moves
            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nAvailable moves for Player 1:")?;
                for move_str in &p1_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
                writeln!(file, "\nAvailable moves for Player 2:")?;
                for move_str in &p2_move_strings {
                    writeln!(file, "  {}", move_str)?;
                }
            }

            // Create state clones and get player references before parallel execution
            let mut state_clone1 = self.state.clone();
            let mut state_clone2 = self.state.clone();
            let player1 = &mut self.player1;
            let player2 = &mut self.player2;

            // Parallel move selection
            let current_turn = self.turn_count;

            let (p1_choice, p2_choice) = rayon::join(
                || player1.choose_move(&mut state_clone1, true, current_turn),
                || player2.choose_move(&mut state_clone2, false, current_turn),
            );

            // Log chosen moves
            let p1_choice_str = self.format_move(&p1_choice.chosen_move, true);
            let p2_choice_str = self.format_move(&p2_choice.chosen_move, false);

            // Update move counters (only count non-None moves)
            if !matches!(p1_choice.chosen_move, MoveChoice::None) {
                self.player1_move_count += 1;
            }
            if !matches!(p2_choice.chosen_move, MoveChoice::None) {
                self.player2_move_count += 1;
            }

            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nPlayer 1 chose: {}\nPlayer 2 chose: {}",
                    p1_choice_str, p2_choice_str
                )?;
            }

            // Generate move instructions
            let move_instructions = generate_instructions_from_move_pair(
                &mut self.state,
                &p1_choice.chosen_move,
                &p2_choice.chosen_move,
                true,
            );

            if let Some(file) = &mut self.log_file {
                writeln!(
                    file,
                    "\nGenerated {} instruction sets",
                    move_instructions.len()
                )?;

                for (i, instructions) in move_instructions.iter().enumerate() {
                    writeln!(
                        file,
                        "\nPossible instruction set {} (probability {:.2}%):",
                        i + 1,
                        instructions.percentage
                    )?;
                    for (j, instruction) in instructions.instruction_list.iter().enumerate() {
                        writeln!(file, "  {}: {:?}", j + 1, instruction)?;
                    }
                }
            }

            // Sample and apply one instruction set based on probabilities
            if let Some(chosen_instructions) = self.sample_instructions(move_instructions) {
                if let Some(file) = &mut self.log_file {
                    writeln!(
                        file,
                        "\nChose instruction set with probability {:.2}%:",
                        chosen_instructions.percentage
                    )?;
                    writeln!(
                        file,
                        "Applying {} instructions:",
                        chosen_instructions.instruction_list.len()
                    )?;
                }

                // Apply all instructions together to maintain proper state
                self.state
                    .apply_instructions(&chosen_instructions.instruction_list);

                if let Some(file) = &mut self.log_file {
                    for (i, instruction) in chosen_instructions.instruction_list.iter().enumerate()
                    {
                        writeln!(file, "Instruction {}: {:?}", i + 1, instruction)?;
                    }
                    self.log_state("State after applying instructions:")?;
                }

                // Reserialize to ensure clean state
                self.state = State::deserialize(self.state.serialize().as_str());
            } else {
                self.log("\nWARNING: No valid instructions generated!")?;
            }

            if let Some(file) = &mut self.log_file {
                writeln!(file, "\nBattle status: {}", self.state.battle_is_over())?;
                file.flush()?;
            }
        }

        let result = self.state.battle_is_over();
        let duration = self.start_time.elapsed();

        self.log(&format!("\nBattle finished with result: {}", result))?;

        // Create and return evaluation result
        let eval_result = EvaluationResult {
            winner: result as i8,
            turns: self.turn_count,
            duration_ms: duration.as_millis() as u64,
            player1_moves: self.player1_move_count,
            player2_moves: self.player2_move_count,
        };

        Ok(eval_result)
    }
}
